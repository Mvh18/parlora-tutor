"""The text-to-speech benchmark with a fake synthesiser and a fake clock."""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from scripts.bench import tts
from scripts.bench.common.schema import validate_result

pytestmark = pytest.mark.unit


class FakeSynth:
    def __init__(self, mode: str) -> None:
        self.oov_mode = mode

    def __call__(self, text: str) -> tuple[float, tuple[str, ...]]:
        return (2.0, ("Zürich",)) if "Zürich" in text else (1.0, ())


def clock(spans: Iterator[float]) -> Callable[[], float]:
    state = {"now": 0.0}
    values = (x for s in spans for x in (s, 0.0))

    def tick() -> float:
        value = state["now"]
        state["now"] = value + next(values)
        return value

    return tick


def test_first_audio_rtf_and_oov_are_aggregated() -> None:
    sentences = ["Hi there.", "Welcome to Zürich."]
    # load 4.0, cold 0.9, then two runs each: 0.3, 0.3 and 0.6, 0.6
    spans = iter([4.0, 0.9, 0.3, 0.3, 0.6, 0.6])

    m = tts.measure(lambda: FakeSynth("cli"), sentences, runs=2, clock=clock(spans))

    assert m["load_s"] == 4.0
    assert m["first_audio_s"]["cold"] == pytest.approx(0.9)
    assert m["first_audio_s"]["summary"]["max"] == pytest.approx(0.6)
    assert m["rtf"]["summary"]["p50"] == pytest.approx(0.3)
    assert m["oov"] == {"mode": "cli", "count": 1, "words": ["Zürich"]}
    assert m["sentences"][1]["oov_words"] == ["Zürich"]


def test_silent_mode_is_flagged() -> None:
    sentences = ["Hi."]
    m = tts.measure(
        lambda: FakeSynth("silent"), sentences, runs=1, clock=clock(iter([1.0, 0.2, 0.1]))
    )

    checks = {c["name"]: c["status"] for c in tts.checks_for(m)}

    assert checks == {"tts_first_audio": "pass", "tts_rtf": "pass", "tts_oov_mode": "warn"}


def test_result_records_the_oov_mode(tmp_path: Path) -> None:
    code = tts.main(
        ["--runs", "1", "--variant", "native", "--output-dir", str(tmp_path)],
        load=lambda: FakeSynth("silent"),
    )

    assert code == 0
    (path,) = tmp_path.glob("*-tts-native.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    validate_result(result)
    assert result["settings"]["oov_mode"] == "silent"
    assert result["measurements"]["oov"]["words"] == ["Zürich"]


def test_sentence_set_has_out_of_vocabulary_words() -> None:
    assert len(tts.SENTENCES) >= 8
    assert any("Zürich" in s for s in tts.SENTENCES)


def test_runs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        tts.main(["--runs", "0", "--output-dir", str(tmp_path)])
