"""Results are written once, validated and free of personal data."""

import json
from pathlib import Path

import pytest

from scripts.bench.common.privacy import PrivacyContext, PrivacyError
from scripts.bench.common.results import RESULTS_DIR, result_filename, write_result
from scripts.bench.common.schema import SchemaError
from tests.bench.conftest import sample_result

pytestmark = pytest.mark.unit


def test_results_directory_is_in_the_repository() -> None:
    assert RESULTS_DIR.parts[-2:] == ("benchmarks", "results")
    assert (RESULTS_DIR / ".gitkeep").is_file()


def test_file_name_is_utc_timestamp_component_and_variant() -> None:
    assert result_filename(sample_result()) == "20261104T153000Z-llm-qwen3-4b.json"


def test_written_result_round_trips(tmp_path: Path, privacy: PrivacyContext) -> None:
    result = sample_result()

    path = write_result(result, tmp_path / "out", context=privacy)

    assert path == tmp_path / "out" / "20261104T153000Z-llm-qwen3-4b.json"
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert json.loads(text) == result


def test_personal_data_blocks_the_write(tmp_path: Path, privacy: PrivacyContext) -> None:
    result = sample_result()
    result["settings"]["cache"] = "C:\\Users\\ada\\cache"

    with pytest.raises(PrivacyError, match=r"settings\.cache: home directory"):
        write_result(result, tmp_path, context=privacy)
    assert list(tmp_path.iterdir()) == []


def test_invalid_result_blocks_the_write(tmp_path: Path, privacy: PrivacyContext) -> None:
    result = sample_result()
    del result["checks"]

    with pytest.raises(SchemaError):
        write_result(result, tmp_path, context=privacy)
    assert list(tmp_path.iterdir()) == []


def test_existing_result_is_never_overwritten(tmp_path: Path, privacy: PrivacyContext) -> None:
    write_result(sample_result(), tmp_path, context=privacy)

    with pytest.raises(FileExistsError):
        write_result(sample_result(), tmp_path, context=privacy)


def test_current_identity_is_the_default_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        PrivacyContext,
        "current",
        classmethod(lambda cls: PrivacyContext(user_names=("zed",), home_dirs=(), host_names=())),
    )
    result = sample_result()
    result["software"]["by"] = "zed"

    with pytest.raises(PrivacyError):
        write_result(result, tmp_path)
