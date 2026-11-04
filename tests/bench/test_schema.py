"""The versioned result schema and the allow-listed machine block."""

from typing import Any

import pytest

from scripts.bench.common.machine import (
    ALLOWED_FIELDS,
    GpuInfo,
    machine_block,
    parse_cpuinfo,
    parse_meminfo_mb,
    parse_nvidia_smi,
)
from scripts.bench.common.schema import (
    REQUIRED_BLOCKS,
    SCHEMA_VERSION,
    SchemaError,
    slug,
    validate_result,
)
from tests.bench.conftest import sample_result

pytestmark = pytest.mark.unit


def test_result_from_recorded_measurements_validates() -> None:
    result = sample_result()

    validate_result(result)
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["created_at"] == "2026-11-04T15:30:00Z"
    for block in REQUIRED_BLOCKS:
        assert block in result


def test_required_blocks_are_the_documented_ones() -> None:
    assert REQUIRED_BLOCKS == ("machine", "software", "settings", "measurements", "checks")


@pytest.mark.parametrize("block", REQUIRED_BLOCKS)
def test_missing_block_fails(block: str) -> None:
    result = sample_result()
    del result[block]

    with pytest.raises(SchemaError, match=block):
        validate_result(result)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("schema_version", 99, "schema_version"),
        ("schema_version", True, "schema_version"),
        ("component", "gpu", "component"),
        ("variant", "Qwen3:4B", "variant"),
        ("created_at", "yesterday", "created_at"),
        ("software", [], "software"),
        ("settings", "fast", "settings"),
        ("measurements", None, "measurements"),
        ("checks", {}, "checks"),
        ("checks", [{"name": "x", "status": "maybe"}], "status"),
        ("checks", [{"status": "pass"}], "name"),
        ("checks", ["pass"], "checks"),
        ("extra", 1, "unknown key"),
        ("measurements", {"x": float("nan")}, "JSON"),
    ],
)
def test_invalid_values_fail(key: str, value: Any, message: str) -> None:
    result = sample_result()
    result[key] = value

    with pytest.raises(SchemaError, match=message):
        validate_result(result)


def test_every_problem_is_reported_at_once() -> None:
    result = sample_result(component="gpu", variant="")

    with pytest.raises(SchemaError) as error:
        validate_result(result)
    assert "component" in str(error.value)
    assert "variant" in str(error.value)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("qwen3:4b", "qwen3-4b"), ("Native", "native"), ("large-v3 turbo", "large-v3-turbo")],
)
def test_slug(text: str, expected: str) -> None:
    assert slug(text) == expected


def test_slug_of_nothing_is_rejected() -> None:
    with pytest.raises(ValueError, match="slug"):
        slug(":::")


# ── machine block ───────────────────────────────────────────────────────────


def test_machine_allow_list_has_no_identifying_fields() -> None:
    assert set(ALLOWED_FIELDS) == {
        "os",
        "os_version",
        "cpu",
        "ram_total_mb",
        "gpu",
        "vram_total_mb",
        "driver",
        "cuda_runtime",
        "docker_engine",
    }


def test_machine_field_outside_the_allow_list_fails() -> None:
    result = sample_result()
    result["machine"]["hostname"] = "box"

    with pytest.raises(SchemaError, match="hostname"):
        validate_result(result)


def test_machine_values_must_be_scalars() -> None:
    result = sample_result()
    result["machine"]["gpu"] = {"name": "x"}

    with pytest.raises(SchemaError, match="gpu"):
        validate_result(result)


def test_machine_block_fills_every_field() -> None:
    block = machine_block(
        os="Linux",
        os_version="6.6",
        cpu="cpu",
        ram_total_mb=1024,
        gpu=GpuInfo(name="RTX", vram_total_mb=8188, driver="616.56"),
        cuda_runtime="12.8",
        docker_engine="28.2.2",
    )

    assert list(block) == list(ALLOWED_FIELDS)
    assert block["gpu"] == "RTX"
    assert block["vram_total_mb"] == 8188
    assert block["driver"] == "616.56"


def test_machine_block_without_gpu() -> None:
    block = machine_block(os="Linux", os_version="6.6", cpu=None, ram_total_mb=None, gpu=None)

    assert block["gpu"] is None
    assert block["vram_total_mb"] is None
    assert block["docker_engine"] is None


def test_parse_cpuinfo() -> None:
    text = "processor\t: 0\nvendor_id\t: x\nmodel name\t: Example CPU @ 3.0GHz\n"
    assert parse_cpuinfo(text) == "Example CPU @ 3.0GHz"
    assert parse_cpuinfo("processor : 0\n") is None


def test_parse_meminfo() -> None:
    assert parse_meminfo_mb("MemTotal:       16303912 kB\nMemFree: 1 kB\n") == 15921
    assert parse_meminfo_mb("MemFree: 1 kB\n") is None


def test_parse_nvidia_smi() -> None:
    assert parse_nvidia_smi("NVIDIA GeForce RTX 4060, 8188, 616.56\n") == GpuInfo(
        name="NVIDIA GeForce RTX 4060", vram_total_mb=8188, driver="616.56"
    )
    assert parse_nvidia_smi("") is None
    assert parse_nvidia_smi("garbage") is None
