"""The docker command that runs a benchmark in the bench image."""

import pytest

from scripts.bench import container

pytestmark = pytest.mark.unit


def test_repository_is_read_only_and_only_named_folders_are_writable() -> None:
    command = container.run_command(
        "docker",
        "scripts.bench.stt",
        ["--runs", "3"],
        gpu=True,
        writable=["benchmarks/results"],
        engine="28.2.2",
    )

    repo = container.REPO_ROOT
    assert command[:3] == ["docker", "run", "--rm"]
    assert f"{repo}:/app:ro" in command
    assert f"{repo / 'benchmarks/results'}:/app/benchmarks/results" in command
    assert f"{container.CACHE_VOLUME}:/cache" in command
    assert "PARLORA_BENCH_VARIANT=container" in command
    assert "PARLORA_BENCH_DOCKER_ENGINE=28.2.2" in command
    assert command[command.index("--gpus") + 1] == "all"
    assert command[-6:] == ["parlora-bench", "python", "-m", "scripts.bench.stt", "--runs", "3"]


def test_no_ports_and_no_gpu_unless_asked() -> None:
    command = container.run_command(
        "docker", "scripts.bench.tts", [], gpu=False, writable=[], engine=None
    )

    assert "--gpus" not in command
    assert not {"-p", "--publish", "--network"} & set(command)
    assert not any(part.startswith("PARLORA_BENCH_DOCKER_ENGINE") for part in command)


def test_build_uses_the_bench_target() -> None:
    command = container.build_command("docker")

    assert command[command.index("--target") + 1] == "bench"
    assert command[command.index("--tag") + 1] == container.IMAGE


def test_only_benchmark_modules_run() -> None:
    with pytest.raises(SystemExit):
        container.main(["os"])
