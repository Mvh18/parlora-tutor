"""Parsers of the memory benchmark's probes, from recorded outputs."""

from pathlib import Path

import pytest

from scripts.bench.common.probes import (
    parse_compute_apps,
    parse_docker_stats,
    parse_ps_rss_mb,
    parse_tasklist_mb,
    size_mb,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "probes"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("text", "mib"),
    [
        ("185.3MiB", 185.3),
        ("7.5GiB", 7680.0),
        ("512KiB", 0.5),
        ("1.5GB", 1536.0),
        ("1048576B", 1.0),
    ],
)
def test_size_mb(text: str, mib: float) -> None:
    assert size_mb(text) == pytest.approx(mib)


@pytest.mark.parametrize("text", ["--", "", "12 parsecs", "MiB"])
def test_size_mb_of_non_sizes(text: str) -> None:
    assert size_mb(text) is None


def test_docker_stats_keeps_only_the_compose_project() -> None:
    assert parse_docker_stats(read("docker_stats.txt")) == {
        "parlora-server-1": 412.6,
        "parlora-postgres-1": 38.2,
    }


def test_compute_apps_keep_names_not_paths() -> None:
    assert parse_compute_apps(read("compute_apps.txt")) == [
        {"name": "ShellHost.exe", "used_mb": None},
        {"name": "explorer.exe", "used_mb": None},
        {"name": "python3", "used_mb": 1290},
    ]


def test_compute_apps_skip_malformed_lines() -> None:
    assert parse_compute_apps("No running processes found\n") == []


def test_tasklist_sums_the_working_sets() -> None:
    assert parse_tasklist_mb(read("tasklist.txt")) == pytest.approx(
        (27_388 + 1_023_424) / 1024, abs=0.1
    )
    assert parse_tasklist_mb("INFO: No tasks are running which match the specified criteria.") == 0


def test_ps_rss_sums_kib() -> None:
    assert parse_ps_rss_mb(" 2048\n1024\n") == 3.0
    assert parse_ps_rss_mb("") == 0
