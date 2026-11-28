"""The smoke call: every synthetic fixture as a learner turn of a real call.

Needs the GPU and Ollama on the host: never in CI.
"""

import json
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.bench import call
from scripts.bench.common.schema import validate_result
from scripts.bench.host import OLLAMA

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def test_every_turn_gets_bot_audio_within_the_latency_target(tmp_path: Path) -> None:
    try:
        urllib.request.urlopen(f"{OLLAMA}/api/version", timeout=2).close()  # noqa: S310 - fixed local URL
    except (urllib.error.URLError, OSError):
        pytest.skip("Ollama is not running on the host")

    code = call.main(["--output-dir", str(tmp_path)])

    (path,) = tmp_path.glob("*-call-native.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    checks = {c["name"]: c["status"] for c in result["checks"]}
    assert code == 0
    assert checks == {
        "call_bot_audio_every_turn": "pass",
        "call_model_p50": "pass",
        "call_model_p95": "pass",
    }
