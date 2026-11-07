"""The speech-to-text benchmark with a fake transcriber and a fake clock."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from scripts.bench import stt
from scripts.bench.common.schema import validate_result

pytestmark = pytest.mark.unit


class FakeSampler:
    source = "fake"
    peak_mb: int | None = None

    def __init__(self) -> None:
        self.reads = iter([1_000, 2_200])

    def used_mb(self) -> int:
        return next(self.reads)

    def __enter__(self) -> "FakeSampler":
        self.peak_mb = 2_400
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class Clock:
    """Each call advances by the next step, so every timed span is one step."""

    def __init__(self, steps: Iterator[float]) -> None:
        self.now, self.steps = 0.0, steps

    def __call__(self) -> float:
        value = self.now
        self.now += next(self.steps)
        return value


def fixtures() -> list[stt.Fixture]:
    return [
        stt.Fixture("a1", "A1", "I would like two coffees, please.", "audio-a1", 10.0),
        stt.Fixture("b2", "B2", "The cat sat on the mat.", "audio-b2", 5.0),
    ]


def hypotheses(audio: str) -> str:
    return {"audio-a1": "i would like 2 coffees please", "audio-b2": "the bat sat on mat"}[audio]


def test_load_first_and_warm_runs_are_measured_apart() -> None:
    # load 3.0 s, first 1.5 s, then warm runs of 0.7 s (a1) and 0.5 s (b2)
    steps = iter([3.0, 1.5, 0.7, 0.7, 0.7, 0.5, 0.5, 0.5])
    clock = Clock(iter(x for s in steps for x in (s, 0.0)))

    m = stt.measure(lambda: hypotheses, fixtures(), runs=3, sampler=FakeSampler(), clock=clock)

    assert m["load_s"] == 3.0
    assert m["first_transcription_s"] == 1.5
    assert m["latency_s"]["cold"] == 1.5
    assert m["latency_s"]["summary"]["count"] == 6
    a1, b2 = m["fixtures"]
    assert a1["latency_s"]["summary"]["p50"] == pytest.approx(0.7)
    assert a1["rtf_p50"] == pytest.approx(0.07)
    assert a1["wer"] == 0.0
    assert b2["wer"] == pytest.approx(2 / 6)
    assert m["wer"] == {"overall": pytest.approx(2 / 12), "label": "synthetic speech"}
    assert m["vram_mb"] == {"baseline": 1_000, "after_load": 1_200, "peak": 1_400}


@pytest.fixture
def no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(stt, "load_fixtures", lambda directory: fixtures())


@pytest.mark.usefixtures("no_audio")
def test_result_records_the_variant_and_the_budget(tmp_path: Path) -> None:
    code = stt.main(
        ["--runs", "2", "--variant", "container", "--output-dir", str(tmp_path)],
        load=lambda: hypotheses,
        sampler=FakeSampler(),
    )

    assert code == 0
    (path,) = tmp_path.glob("*-stt-container.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    assert result["settings"]["compute_type"] == "int8"
    assert result["settings"]["beam_size"] == 5
    assert result["checks"][0]["name"] == "stt_warm_p95"


@pytest.mark.usefixtures("no_audio")
def test_cuda_failure_is_recorded_instead_of_crashing(tmp_path: Path) -> None:
    def broken() -> Any:
        raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")

    code = stt.main(
        ["--variant", "native", "--output-dir", str(tmp_path)], load=broken, sampler=FakeSampler()
    )

    assert code == 1
    (path,) = tmp_path.glob("*-stt-native.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    assert "cublas64_12.dll" in result["measurements"]["error"]
    assert result["checks"][0]["status"] == "fail"


def test_variant_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(stt.VARIANT_ENV, "container")
    assert stt.detect_variant() == "container"


def test_runs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        stt.main(["--runs", "0", "--output-dir", str(tmp_path)])


def test_committed_fixtures_load_as_float_pcm() -> None:
    pytest.importorskip("numpy")

    loaded = stt.load_fixtures()

    assert len(loaded) >= 10
    assert all(3.0 <= f.duration_s <= 15.0 for f in loaded)
    assert loaded[0].audio.dtype.name == "float32"
    assert float(abs(loaded[0].audio).max()) <= 1.0
