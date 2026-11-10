"""faster-whisper speech-to-text adapter (ADR 0002).

One learner turn arrives as 16 kHz 16-bit PCM, the rate the call transport records
at. It is converted in memory to the float32 samples faster-whisper takes (stereo
mixed down to mono) and transcribed off the event loop, in the language the call
was set up with. Calls are serialised: one model instance serves one call at a time.
Model weights are downloaded into the Hugging Face cache (the models volume in the
server image) on first use, never stored in the repository.
"""

import asyncio
import threading
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from parlora_core.audio import AudioChunk, Transcript
from parlora_core.errors import InvalidAudio
from parlora_core.languages import LanguageCode
from parlora_server.adapters.cuda import add_cuda_dll_directories
from parlora_server.config import SttSettings

SAMPLE_RATE = 16_000


class WhisperModel(Protocol):
    def transcribe(self, audio: Any, **options: Any) -> tuple[Iterable[Any], Any]: ...


def to_float32(audio: AudioChunk) -> Any:
    """16-bit PCM as float32 samples in [-1, 1), mixed down to mono."""
    import numpy as np

    samples = np.frombuffer(audio.pcm, dtype="<i2").astype(np.float32) / 32768.0
    if audio.channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1, dtype=np.float32)
    return samples


class WhisperSpeechToText:
    """`parlora_core.ports.SpeechToText` over a faster-whisper model."""

    def __init__(
        self,
        model: WhisperModel,
        *,
        beam_size: int = 5,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model = model
        self.beam_size = beam_size
        self._clock = clock
        self._lock = threading.Lock()

    async def transcribe(self, audio: AudioChunk, *, language: LanguageCode) -> Transcript:
        if audio.sample_rate != SAMPLE_RATE:
            raise InvalidAudio(
                f"speech-to-text takes {SAMPLE_RATE} Hz audio, got {audio.sample_rate} Hz"
            )
        if not audio.pcm:
            return Transcript("", language, 0.0, 0.0)
        samples = to_float32(audio)
        started = self._clock()
        text = await asyncio.to_thread(self._transcribe, samples, language)
        return Transcript(text, language, audio.duration_s, self._clock() - started)

    def _transcribe(self, samples: Any, language: LanguageCode) -> str:
        with self._lock:
            # The segments are a generator: decoding happens while it is consumed.
            segments, _ = self._model.transcribe(
                samples,
                language=str(language),
                beam_size=self.beam_size,
                condition_on_previous_text=False,
            )
            return "".join(segment.text for segment in segments).strip()


def _faster_whisper(name: str, **options: Any) -> WhisperModel:
    from faster_whisper import WhisperModel as Model

    model: WhisperModel = Model(name, **options)
    return model


def load_whisper(
    settings: SttSettings, *, factory: Callable[..., WhisperModel] = _faster_whisper
) -> WhisperSpeechToText:
    """Load the model the settings name (downloading it on first use)."""
    add_cuda_dll_directories()
    model = factory(settings.model, device=settings.device, compute_type=settings.compute_type)
    return WhisperSpeechToText(model, beam_size=settings.beam_size)
