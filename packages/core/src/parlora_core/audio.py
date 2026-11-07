"""Audio and transcript values passed through the speech ports."""

from dataclasses import dataclass

from parlora_core.errors import InvalidAudio
from parlora_core.languages import LanguageCode

MIN_SAMPLE_RATE = 8_000
MAX_SAMPLE_RATE = 192_000
SAMPLE_WIDTH = 2  # bytes per sample: 16-bit signed PCM


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Interleaved 16-bit little-endian PCM at `sample_rate` with 1 or 2 channels."""

    pcm: bytes
    sample_rate: int
    channels: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.pcm, bytes):
            raise InvalidAudio("audio must be bytes of 16-bit PCM")
        if not MIN_SAMPLE_RATE <= self.sample_rate <= MAX_SAMPLE_RATE:
            raise InvalidAudio(
                f"sample rate {self.sample_rate} is outside {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE} Hz"
            )
        if self.channels not in (1, 2):
            raise InvalidAudio(f"{self.channels} channels; only mono or stereo is supported")
        if len(self.pcm) % (SAMPLE_WIDTH * self.channels):
            raise InvalidAudio(
                f"{len(self.pcm)} bytes is not a whole number of 16-bit frames "
                f"for {self.channels} channel(s)"
            )

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / (SAMPLE_WIDTH * self.channels * self.sample_rate)


@dataclass(frozen=True, slots=True)
class Transcript:
    """What the learner said, with the audio length and the time it took to transcribe."""

    text: str
    language: LanguageCode
    audio_s: float
    processing_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise InvalidAudio("a transcript's text must be a string")
        if self.audio_s < 0 or self.processing_s < 0:
            raise InvalidAudio("transcript timings must not be negative")

    @property
    def real_time_factor(self) -> float | None:
        """Processing time over audio time; `None` for empty audio."""
        return self.processing_s / self.audio_s if self.audio_s else None
