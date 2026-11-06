"""One real LLM run through the host's Ollama. Needs the GPU and Ollama: never in CI."""

import json
from pathlib import Path

import pytest

from scripts.bench import llm
from scripts.bench.common.schema import validate_result
from scripts.bench.host import ollama_version

pytestmark = pytest.mark.gpu

MODEL = "qwen3:4b-instruct-2507-q4_K_M"


def test_one_warm_run_writes_a_valid_result(tmp_path: Path) -> None:
    if ollama_version() is None:
        pytest.skip("Ollama is not running on the host")

    assert llm.main([MODEL, "--runs", "1", "--output-dir", str(tmp_path)]) == 0

    (path,) = tmp_path.glob("*-llm-qwen3-4b-instruct-2507-q4_k_m.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    assert result["measurements"]["tokens_per_s"]["summary"]["count"] == 1
    assert result["machine"]["gpu"]
