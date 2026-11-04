"""The `machine` block of a result: an allow-list of hardware and runtime facts.

Only the fields in `ALLOWED_FIELDS` may appear. Host names, user names, file system
paths and IP addresses are never collected. The runners probe the host; this
module only shapes the block and parses the probes' output.
"""

from dataclasses import dataclass

ALLOWED_FIELDS = (
    "os",
    "os_version",
    "cpu",
    "ram_total_mb",
    "gpu",
    "vram_total_mb",
    "driver",
    "cuda_runtime",
    "docker_engine",
)

MachineValue = str | int | None


@dataclass(frozen=True)
class GpuInfo:
    name: str
    vram_total_mb: int
    driver: str


def machine_block(
    *,
    os: str,
    os_version: str,
    cpu: str | None,
    ram_total_mb: int | None,
    gpu: GpuInfo | None,
    cuda_runtime: str | None = None,
    docker_engine: str | None = None,
) -> dict[str, MachineValue]:
    """Every allowed field, in the documented order; unknown values are `None`."""
    return {
        "os": os,
        "os_version": os_version,
        "cpu": cpu,
        "ram_total_mb": ram_total_mb,
        "gpu": gpu.name if gpu else None,
        "vram_total_mb": gpu.vram_total_mb if gpu else None,
        "driver": gpu.driver if gpu else None,
        "cuda_runtime": cuda_runtime,
        "docker_engine": docker_engine,
    }


def machine_problems(machine: object) -> list[str]:
    """Why `machine` breaks the allow-list; empty when it is valid."""
    if not isinstance(machine, dict):
        return ["machine must be an object"]
    problems = [
        f"machine.{key} is not an allowed field" for key in machine if key not in ALLOWED_FIELDS
    ]
    problems += [
        f"machine.{key} must be a string, an integer or null"
        for key, value in machine.items()
        if key in ALLOWED_FIELDS
        and not (value is None or (isinstance(value, str | int) and not isinstance(value, bool)))
    ]
    return problems


def parse_cpuinfo(text: str) -> str | None:
    """The CPU model from Linux `/proc/cpuinfo`."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return None


def parse_meminfo_mb(text: str) -> int | None:
    """Total RAM in MiB from Linux `/proc/meminfo` (which reports kB)."""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "MemTotal":
            return int(value.split()[0]) // 1024
    return None


def parse_nvidia_smi(text: str) -> GpuInfo | None:
    """The first GPU from `nvidia-smi --query-gpu=name,memory.total,driver_version
    --format=csv,noheader,nounits`."""
    first = text.strip().splitlines()[0] if text.strip() else ""
    parts = [part.strip() for part in first.split(",")]
    if len(parts) != 3 or not parts[1].isdigit():
        return None
    return GpuInfo(name=parts[0], vram_total_mb=int(parts[1]), driver=parts[2])
