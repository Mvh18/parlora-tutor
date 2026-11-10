"""The memory benchmark's VRAM rule with fake loaders and a fake sampler."""

from collections.abc import Iterator
from typing import Any

import pytest

from scripts.bench import memory

pytestmark = pytest.mark.unit


class Sampler:
    """`used_mb` returns the next scripted reading; peaks come from a script too."""

    def __init__(self, readings: list[int], peaks: list[int]) -> None:
        self.readings: Iterator[int] = iter(readings)
        self.peaks: Iterator[int] = iter(peaks)
        self.peak_mb: int | None = None

    def used_mb(self) -> int:
        return next(self.readings)

    def __enter__(self) -> "Sampler":
        return self

    def __exit__(self, *exc: object) -> None:
        self.peak_mb = next(self.peaks)


def run(sampler: Sampler, models: list[str], limit: int = 7_680) -> Any:
    calls: list[str] = []

    def transcribe(audio: Any) -> str:
        calls.append("transcribe")
        return "text"

    def load_llm(model: str) -> int:
        calls.append(f"load {model}")
        return 3_000

    result = memory.measure(
        load_stt=lambda: transcribe,
        stt_audio="pcm",
        load_llm=load_llm,
        unload_all=lambda: calls.append("unload"),
        models=models,
        sampler=sampler,
        limit_mb=limit,
        settle=lambda: None,
    )
    return result, calls


def test_each_model_is_checked_with_stt_loaded_and_the_baseline_subtracted() -> None:
    # baseline 1000, with STT 2300; qwen3:4b reaches 5400, qwen3:8b reaches 8900
    sampler = Sampler(readings=[1_000, 2_300, 5_300, 8_800], peaks=[5_400, 8_900])

    (measurements, checks), calls = run(sampler, ["qwen3:4b", "qwen3:8b"])

    assert calls == ["unload", "transcribe", "load qwen3:4b", "unload", "load qwen3:8b", "unload"]
    assert measurements["vram_mb"] == {
        "baseline": 1_000,
        "stt_above_baseline": 1_300,
        "limit": 7_680,
    }
    small, large = measurements["models"]
    assert small == {
        "model": "qwen3:4b",
        "used_mb": 5_400,
        "above_baseline_mb": 4_400,
        "llm_mb": 3_100,
        "ollama_size_vram_mb": 3_000,
        "status": "pass",
    }
    assert large["above_baseline_mb"] == 7_900
    assert [c["status"] for c in checks] == ["pass", "fail"]
    assert memory.exit_code_of(checks) == 1


def test_all_within_the_limit_exits_zero() -> None:
    sampler = Sampler(readings=[1_000, 2_300, 5_300], peaks=[5_400])

    (_, checks), _ = run(sampler, ["gemma3:4b"])

    assert memory.exit_code_of(checks) == 0


def test_a_lower_limit_turns_a_pass_into_a_fail() -> None:
    sampler = Sampler(readings=[1_000, 2_300, 5_300], peaks=[5_400])

    (_, checks), _ = run(sampler, ["gemma3:4b"], limit=4_000)

    assert checks[0]["status"] == "fail"
    assert checks[0]["limit"] == 4_000


def test_kokoro_is_loaded_after_stt_and_counted_with_the_speech_models() -> None:
    # baseline 1000, with STT 2300, with Kokoro 2900; qwen3:4b reaches 6000
    sampler = Sampler(readings=[1_000, 2_300, 2_900, 5_900], peaks=[6_000])
    calls: list[str] = []

    def load_tts() -> Any:
        calls.append("load kokoro")
        return lambda text: calls.append(f"speak {text}")

    measurements, checks = memory.measure(
        load_stt=lambda: lambda audio: calls.append("transcribe"),
        stt_audio="pcm",
        load_tts=load_tts,
        load_llm=lambda model: calls.append(f"load {model}"),
        unload_all=lambda: calls.append("unload"),
        models=["qwen3:4b"],
        sampler=sampler,
        limit_mb=7_680,
        settle=lambda: None,
    )

    assert calls[:4] == ["unload", "transcribe", "load kokoro", f"speak {memory.TTS_SENTENCE}"]
    assert measurements["vram_mb"]["tts_above_stt"] == 600
    (entry,) = measurements["models"]
    assert entry["above_baseline_mb"] == 5_000
    assert entry["llm_mb"] == 3_100
    assert checks[0]["status"] == "pass"


def test_without_kokoro_the_tts_share_is_null() -> None:
    sampler = Sampler(readings=[1_000, 2_300, 5_300], peaks=[5_400])

    (measurements, _), _ = run(sampler, ["gemma3:4b"])

    assert measurements["vram_mb"].get("tts_above_stt") is None
