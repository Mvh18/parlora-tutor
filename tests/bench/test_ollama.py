"""Timings parsed from Ollama's streamed chat responses and from /api/ps."""

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.bench.common.ollama import (
    StreamTimings,
    model_size_vram_mb,
    model_software,
    parse_stream,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).parent / "fixtures" / "ollama"


def events(name: str) -> list[tuple[float, dict[str, Any]]]:
    lines = (FIXTURES / name).read_text(encoding="utf-8").splitlines()
    return [(row["t"], row["chunk"]) for row in map(json.loads, lines)]


def test_final_chunk_timings_are_converted_to_seconds() -> None:
    timings = parse_stream(events("chat-stream.jsonl"))

    assert timings == StreamTimings(
        ttft_s=0.31,
        total_s=2.35,
        load_s=0.015,
        prompt_eval_s=0.25,
        prompt_eval_count=96,
        eval_s=2.0,
        eval_count=128,
        tokens_per_s=64.0,
        done_reason="length",
    )


def test_time_to_first_token_skips_empty_chunks() -> None:
    assert parse_stream(events("chat-stream.jsonl")).ttft_s == 0.31


def test_cold_stream_reports_its_load_duration() -> None:
    timings = parse_stream(events("chat-stream-cold.jsonl"))

    assert timings.load_s == 3.5
    assert timings.ttft_s == 3.92


def test_generate_style_chunks_are_understood() -> None:
    stream = [
        (0.2, {"response": "", "done": False}),
        (0.4, {"response": "Hi", "done": False}),
        (1.0, {"response": "", "done": True, "eval_count": 10, "eval_duration": 500_000_000}),
    ]

    timings = parse_stream(stream)

    assert timings.ttft_s == 0.4
    assert timings.tokens_per_s == 20.0
    assert timings.load_s == 0.0


def test_stream_without_a_token_fails() -> None:
    stream = [(0.1, {"message": {"content": ""}, "done": True, "eval_count": 0})]

    with pytest.raises(ValueError, match="no token"):
        parse_stream(stream)


def test_stream_without_a_final_chunk_fails() -> None:
    with pytest.raises(ValueError, match="final chunk"):
        parse_stream(events("chat-stream.jsonl")[:-1])


def test_error_chunk_is_reported() -> None:
    with pytest.raises(ValueError, match="model 'x' not found"):
        parse_stream([(0.1, {"error": "model 'x' not found"})])


def test_zero_eval_duration_fails() -> None:
    stream = [
        (0.1, {"message": {"content": "a"}, "done": False}),
        (0.2, {"message": {"content": ""}, "done": True, "eval_count": 1, "eval_duration": 0}),
    ]

    with pytest.raises(ValueError, match="eval_duration"):
        parse_stream(stream)


def test_timings_serialise() -> None:
    data = parse_stream(events("chat-stream.jsonl")).to_dict()

    assert data["tokens_per_s"] == 64.0
    assert set(data) == {
        "ttft_s",
        "total_s",
        "load_s",
        "prompt_eval_s",
        "prompt_eval_count",
        "eval_s",
        "eval_count",
        "tokens_per_s",
        "done_reason",
    }


def test_size_vram_of_a_loaded_model_in_megabytes() -> None:
    ps = json.loads((FIXTURES / "ps.json").read_text(encoding="utf-8"))

    assert model_size_vram_mb(ps, "qwen3:4b") == 3500
    assert model_size_vram_mb(ps, "gemma3:4b") is None


def test_model_software_from_the_tags_listing() -> None:
    tags = {
        "models": [
            {
                "name": "qwen3:4b",
                "digest": "e55aed6fe643" + "0" * 52,
                "size": 2_620_000_000,
                "details": {
                    "family": "qwen3",
                    "parameter_size": "4.0B",
                    "quantization_level": "Q4_K_M",
                },
            }
        ]
    }

    assert model_software(tags, "qwen3:4b") == {
        "model": "qwen3:4b",
        "digest": "e55aed6fe643" + "0" * 52,
        "size_mb": 2498,
        "parameter_size": "4.0B",
        "quantization": "Q4_K_M",
    }


def test_model_missing_from_the_tags_listing_fails() -> None:
    with pytest.raises(LookupError, match="ollama pull gemma3:4b"):
        model_software({"models": []}, "gemma3:4b")
