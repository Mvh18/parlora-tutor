"""One real speech-to-text run on the GPU. Needs the bench group and a GPU: never in CI."""

import json
from pathlib import Path

import pytest

from scripts.bench import stt
from scripts.bench.common.schema import validate_result

pytestmark = pytest.mark.gpu


def test_one_warm_run_per_fixture_writes_a_valid_result(tmp_path: Path) -> None:
    pytest.importorskip("faster_whisper")

    code = stt.main(["--runs", "1", "--variant", "native", "--output-dir", str(tmp_path)])

    (path,) = tmp_path.glob("*-stt-native.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    assert code == 0, result["measurements"].get("error")
    assert result["measurements"]["wer"]["overall"] < 0.2
