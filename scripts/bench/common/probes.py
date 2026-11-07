"""Parse the outputs of the memory benchmark's probes (docker, nvidia-smi, tasklist, ps).

Only names and sizes are kept. Process paths are reduced to the executable's name,
because a path can hold the user name.
"""

import re
from pathlib import PureWindowsPath

STACK_PREFIX = "parlora-"
_UNITS = {
    "b": 1 / 1024**2,
    "kib": 1 / 1024,
    "mib": 1.0,
    "gib": 1024.0,
    "kb": 1 / 1024,
    "mb": 1.0,
    "gb": 1024.0,
}


def size_mb(text: str) -> float | None:
    """`185.3MiB`, `7.691GiB`, `512kB` -> MiB; `None` when it is not a size."""
    match = re.fullmatch(r"\s*([\d.]+)\s*([A-Za-z]+)\s*", text)
    if not match or match.group(2).lower() not in _UNITS:
        return None
    return float(match.group(1)) * _UNITS[match.group(2).lower()]


def parse_docker_stats(text: str, prefix: str = STACK_PREFIX) -> dict[str, float]:
    """`docker stats --no-stream --format "{{.Name}}\\t{{.MemUsage}}"` -> MiB used per
    container of the Compose project (names starting with `prefix`)."""
    usage = {}
    for line in text.splitlines():
        name, _, mem = line.partition("\t")
        used = size_mb(mem.split("/")[0]) if mem else None
        if name.startswith(prefix) and used is not None:
            usage[name.strip()] = round(used, 1)
    return usage


def parse_compute_apps(text: str) -> list[dict[str, int | str | None]]:
    """`nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`
    -> executable names (never paths) with their VRAM in MiB when the driver reports it."""
    apps: list[dict[str, int | str | None]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        name = PureWindowsPath(parts[1]).name or parts[1]
        memory = parts[2].removesuffix("MiB").strip()
        apps.append({"name": name, "used_mb": int(memory) if memory.isdigit() else None})
    return apps


def parse_tasklist_mb(text: str) -> float:
    """`tasklist /FO CSV /NH` rows -> total working set in MiB (`"27,388 KB"` columns)."""
    total_kb = 0
    for line in text.splitlines():
        fields = re.findall(r'"([^"]*)"', line)
        if len(fields) >= 5:
            digits = re.sub(r"\D", "", fields[4])
            total_kb += int(digits) if digits else 0
    return round(total_kb / 1024, 1)


def parse_ps_rss_mb(text: str) -> float:
    """`ps -o rss=` lines (KiB) -> total MiB."""
    return round(sum(int(v) for v in text.split() if v.isdigit()) / 1024, 1)
