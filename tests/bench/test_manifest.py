"""The synthetic fixture manifest and its WAV files, checked against temporary files."""

import wave
from pathlib import Path
from typing import Any

import pytest

from scripts.bench.common.manifest import MAX_BYTES, manifest_problems

pytestmark = pytest.mark.unit

BANDS = ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2", "C1", "C1"]


def write_wav(
    path: Path, seconds: float = 4.0, rate: int = 16_000, channels: int = 1, width: int = 2
) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(b"\0" * int(seconds * rate) * channels * width)


def manifest(count: int = 10) -> dict[str, Any]:
    return {
        "generator": {
            "kokoro_model_sha256": "a" * 64,
            "voices_sha256": "b" * 64,
            "packages": {"misaki": "0.9.4"},
            "oov_mode": "cli",
        },
        "utterances": [
            {"id": f"u{i}", "cefr": BANDS[i % 10], "voice": "af_heart", "text": f"Sentence {i}."}
            for i in range(count)
        ],
    }


@pytest.fixture
def fixtures(tmp_path: Path) -> Path:
    for i in range(10):
        write_wav(tmp_path / f"u{i}.wav")
    return tmp_path


def test_valid_manifest_and_audio(fixtures: Path) -> None:
    assert manifest_problems(manifest(), fixtures) == []


def test_text_only_manifest_before_generation(tmp_path: Path) -> None:
    data = manifest()
    del data["generator"]

    assert manifest_problems(data, tmp_path, generated=False) == []


def test_missing_wav_is_reported(fixtures: Path) -> None:
    (fixtures / "u3.wav").unlink()

    assert manifest_problems(manifest(), fixtures) == ["u3.wav is missing"]


def test_wav_without_entry_is_reported(fixtures: Path) -> None:
    write_wav(fixtures / "stray.wav")

    assert manifest_problems(manifest(), fixtures) == ["stray.wav has no manifest entry"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 24_000}, "24000 Hz"),
        ({"channels": 2}, "2 channel(s)"),
        ({"width": 4}, "32-bit"),
        ({"seconds": 1.0}, "lasts 1.00 s"),
        ({"seconds": 15.5}, "lasts 15.50 s"),
    ],
)
def test_wrong_audio_format_is_reported(
    fixtures: Path, kwargs: dict[str, Any], message: str
) -> None:
    write_wav(fixtures / "u0.wav", **kwargs)

    problems = manifest_problems(manifest(), fixtures)

    assert len(problems) == 1
    assert message in problems[0]


def test_large_wav_is_reported(fixtures: Path) -> None:
    (fixtures / "u0.wav").write_bytes(b"\0" * MAX_BYTES)

    problems = manifest_problems(manifest(), fixtures)

    assert any("the limit is" in p for p in problems)
    assert any("not a PCM WAV" in p for p in problems)


def test_too_few_utterances_and_bands(tmp_path: Path) -> None:
    data = manifest(count=4)
    for entry in data["utterances"]:
        entry["cefr"] = "A1"

    problems = manifest_problems(data, tmp_path, generated=False)

    assert "4 utterances, at least 10 are required" in problems
    assert "1 CEFR bands, at least 3 are required" in problems


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("id", "Bad Id", "lowercase slug"),
        ("id", "u1", "repeated"),
        ("cefr", "D1", "cefr"),
        ("text", " ", "text"),
        ("voice", None, "voice"),
    ],
)
def test_invalid_entry_is_reported(field: str, value: Any, message: str, tmp_path: Path) -> None:
    data = manifest()
    data["utterances"][0][field] = value

    problems = manifest_problems(data, tmp_path, generated=False)

    assert len(problems) == 1
    assert message in problems[0]


def test_entries_must_be_objects(tmp_path: Path) -> None:
    data = manifest()
    data["utterances"][0] = "u0"

    assert "utterances[0] must be an object" in manifest_problems(data, tmp_path, generated=False)


def test_utterances_must_be_a_list(tmp_path: Path) -> None:
    assert manifest_problems({"utterances": {}}, tmp_path) == ["utterances must be a list"]


def test_generator_is_required_once_generated(fixtures: Path) -> None:
    data = manifest()
    data["generator"]["oov_mode"] = ""
    assert manifest_problems(data, fixtures) == ["generator.oov_mode is missing"]

    del data["generator"]
    assert manifest_problems(data, fixtures) == [
        "generator must be an object once the audio is generated"
    ]


def test_committed_fixtures_are_valid() -> None:
    import json

    from scripts.bench.fixtures import FIXTURES

    data = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))

    assert manifest_problems(data, FIXTURES) == []
    assert data["generator"]["oov_mode"] == "cli", "fixtures must be generated in the bench image"
