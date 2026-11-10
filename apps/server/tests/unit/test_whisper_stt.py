"""The faster-whisper speech-to-text adapter with a fake model."""

import asyncio
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from parlora_core.audio import AudioChunk, Transcript
from parlora_core.errors import InvalidAudio
from parlora_core.languages import LanguageCode
from parlora_core.ports import SpeechToText
from parlora_server.adapters import cuda
from parlora_server.adapters.stt import whisper
from parlora_server.config import SttSettings

pytestmark = pytest.mark.unit

EN = LanguageCode("en")


class FakeModel:
    """Records each call and returns its segments as a generator, like faster-whisper."""

    def __init__(self, *texts: str) -> None:
        self.texts = texts
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def transcribe(self, audio: Any, **options: Any) -> tuple[Iterator[Any], Any]:
        self.calls.append((audio, options))
        segments = (SimpleNamespace(text=text) for text in self.texts)
        return segments, SimpleNamespace(language=options.get("language"))


def pcm(*samples: int, channels: int = 1, rate: int = 16_000) -> AudioChunk:
    return AudioChunk(np.array(samples, dtype="<i2").tobytes(), rate, channels)


def ticking(*times: float) -> Any:
    values = iter(times)
    return lambda: next(values)


def transcribe(adapter: whisper.WhisperSpeechToText, audio: AudioChunk) -> Transcript:
    return asyncio.run(adapter.transcribe(audio, language=EN))


def test_implements_the_port() -> None:
    adapter: SpeechToText = whisper.WhisperSpeechToText(FakeModel())
    assert adapter is not None


def test_returns_the_model_text_with_the_language_and_timings() -> None:
    model = FakeModel(" I went to the market", " yesterday. ")
    adapter = whisper.WhisperSpeechToText(model, beam_size=3, clock=ticking(10.0, 10.25))
    audio = pcm(*([0] * 8_000))

    transcript = transcribe(adapter, audio)

    assert transcript.text == "I went to the market yesterday."
    assert transcript.language == EN
    assert transcript.audio_s == pytest.approx(0.5)
    assert transcript.processing_s == pytest.approx(0.25)
    (_, options) = model.calls[0]
    assert options["language"] == "en"
    assert options["beam_size"] == 3


def test_pcm_becomes_float32_in_minus_one_to_one() -> None:
    model = FakeModel("ok")
    adapter = whisper.WhisperSpeechToText(model)

    transcribe(adapter, pcm(0, 16_384, -32_768, 32_767))

    (audio, _) = model.calls[0]
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, [0.0, 0.5, -1.0, 32_767 / 32_768])


def test_stereo_is_mixed_down_to_mono() -> None:
    model = FakeModel("ok")
    adapter = whisper.WhisperSpeechToText(model)

    transcribe(adapter, pcm(16_384, 0, -16_384, -16_384, channels=2))

    (audio, _) = model.calls[0]
    np.testing.assert_allclose(audio, [0.25, -0.5])


def test_empty_audio_is_not_sent_to_the_model() -> None:
    model = FakeModel("hallucination")
    adapter = whisper.WhisperSpeechToText(model)

    transcript = transcribe(adapter, pcm())

    assert transcript.text == ""
    assert transcript.audio_s == 0
    assert model.calls == []


def test_audio_must_be_16_khz() -> None:
    adapter = whisper.WhisperSpeechToText(FakeModel("ok"))

    with pytest.raises(InvalidAudio, match="16000"):
        transcribe(adapter, pcm(0, 0, rate=48_000))


def test_load_builds_the_model_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(whisper, "add_cuda_dll_directories", lambda: ["cublas"])

    def factory(name: str, **options: Any) -> FakeModel:
        created.append((name, options))
        return FakeModel("ok")

    settings = SttSettings(model="small", device="cpu", compute_type="float32", beam_size=2)
    adapter = whisper.load_whisper(settings, factory=factory)

    assert created == [("small", {"device": "cpu", "compute_type": "float32"})]
    assert adapter.beam_size == 2


def test_cuda_dll_directories_are_only_added_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    added: list[str] = []
    monkeypatch.setattr(cuda, "site_packages", lambda: [str(tmp_path)])
    monkeypatch.setattr(cuda, "add_dll_directory", added.append)
    monkeypatch.setenv("PATH", "before")

    monkeypatch.setattr(sys, "platform", "linux")
    assert cuda.add_cuda_dll_directories() == []

    monkeypatch.setattr(sys, "platform", "win32")
    assert cuda.add_cuda_dll_directories() == ["cublas", "cudnn"]
    assert [Path(a).parent.name for a in added] == ["cublas", "cudnn"]
    assert str(tmp_path / "nvidia" / "cudnn" / "bin") in os.environ["PATH"]
