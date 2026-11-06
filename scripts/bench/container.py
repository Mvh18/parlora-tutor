"""Run a benchmark module in the local bench image (never published).

Builds the `bench` target of docker/server.Dockerfile, then runs the module with
the repository mounted read-only at /app, only the given folders writable (results
or fixtures), the `parlora-bench-cache` volume for weights at /cache, the GPU when
asked, the host's Docker engine version for the result's machine block, and no
published ports.

Usage: python -m scripts.bench.container [--gpu] [--writable DIR ...] MODULE [ARGS ...]
"""

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "parlora-bench"
CACHE_VOLUME = "parlora-bench-cache"


def docker() -> str:
    executable = shutil.which("docker")
    if not executable:
        raise SystemExit("bench: docker is not on PATH; start Docker Desktop")
    return executable


def engine_version(executable: str) -> str | None:
    result = subprocess.run(
        [executable, "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def build_command(executable: str) -> list[str]:
    dockerfile = str(REPO_ROOT / "docker" / "server.Dockerfile")
    return [
        executable,
        "build",
        "--file",
        dockerfile,
        "--target",
        "bench",
        "--tag",
        IMAGE,
        str(REPO_ROOT),
    ]


def run_command(
    executable: str,
    module: str,
    args: Sequence[str],
    *,
    gpu: bool,
    writable: Sequence[str],
    engine: str | None,
    variant: str = "container",
) -> list[str]:
    command = [executable, "run", "--rm", "--volume", f"{REPO_ROOT}:/app:ro"]
    for folder in writable:
        command += ["--volume", f"{REPO_ROOT / folder}:/app/{Path(folder).as_posix()}"]
    command += ["--volume", f"{CACHE_VOLUME}:/cache", "--env", f"PARLORA_BENCH_VARIANT={variant}"]
    if engine:
        command += ["--env", f"PARLORA_BENCH_DOCKER_ENGINE={engine}"]
    if gpu:
        command += ["--gpus", "all"]
    return [*command, IMAGE, "python", "-m", module, *args]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--writable", action="append", default=[])
    parser.add_argument("module")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    if not options.module.startswith("scripts.bench."):
        parser.error("only scripts.bench modules run in the bench image")

    executable = docker()
    built = subprocess.run(build_command(executable), check=False)
    if built.returncode:
        return built.returncode
    command = run_command(
        executable,
        options.module,
        options.args,
        gpu=options.gpu,
        writable=options.writable,
        engine=engine_version(executable),
    )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
