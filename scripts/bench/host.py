"""Probe the host for the `machine` block, and check the benchmark prerequisites.

Only the allow-listed facts of `scripts.bench.common.machine` are collected. Run as
`python -m scripts.bench.host` (what `just bench-setup` does) to print them and
check that the GPU and Ollama answer.
"""

import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from scripts.bench.common.machine import (
    GpuInfo,
    MachineValue,
    machine_block,
    parse_cpuinfo,
    parse_meminfo_mb,
    parse_nvidia_smi,
)

OLLAMA = "http://127.0.0.1:11434"
# The bench recipes pass the host's Docker engine version into the container.
DOCKER_ENGINE_ENV = "PARLORA_BENCH_DOCKER_ENGINE"


def cpu_model() -> str | None:
    if sys.platform == "win32":
        import winreg

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or None
    else:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            return parse_cpuinfo(cpuinfo.read_text(encoding="utf-8"))
        return platform.processor() or None


def ram_total_mb() -> int | None:
    if sys.platform == "win32":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.ullTotalPhys) // (1024 * 1024)
    else:
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            return parse_meminfo_mb(meminfo.read_text(encoding="utf-8"))
        return None


def gpu_info() -> GpuInfo | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    query = "--query-gpu=name,memory.total,driver_version"
    result = subprocess.run(
        [smi, query, "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    return parse_nvidia_smi(result.stdout) if result.returncode == 0 else None


def os_version() -> str:
    # Windows: the build ("10.0.26200"); Linux: the kernel release.
    return platform.version() if sys.platform == "win32" else platform.release()


def collect_machine(cuda_runtime: str | None = None) -> dict[str, MachineValue]:
    return machine_block(
        os=platform.system(),
        os_version=os_version(),
        cpu=cpu_model(),
        ram_total_mb=ram_total_mb(),
        gpu=gpu_info(),
        cuda_runtime=cuda_runtime,
        docker_engine=os.environ.get(DOCKER_ENGINE_ENV) or None,
    )


def ollama_version(endpoint: str = OLLAMA) -> str | None:
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError(f"Ollama endpoint must be http(s), got {endpoint!r}")
    try:
        with urllib.request.urlopen(f"{endpoint}/api/version", timeout=5) as response:  # noqa: S310 - scheme checked above
            return str(json.load(response)["version"])
    except (urllib.error.URLError, OSError, KeyError, ValueError):
        return None


def main() -> int:
    machine = collect_machine()
    print(json.dumps(machine, indent=2))
    problems = []
    if machine["gpu"] is None:
        problems.append("no NVIDIA GPU found (nvidia-smi is missing or failed)")
    version = ollama_version()
    if version is None:
        problems.append(f"Ollama is not answering on {OLLAMA}: start the Ollama app")
    else:
        print(f"Ollama {version} on {OLLAMA}")
    for problem in problems:
        print(f"bench-setup: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
