import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlora_core.audio import AudioChunk, Transcript
from parlora_core.errors import DomainError, InvalidAudio
from parlora_core.languages import LanguageCode

pytestmark = pytest.mark.unit

EN = LanguageCode("en")


def test_audio_chunk_duration() -> None:
    chunk = AudioChunk(pcm=b"\0\0" * 16_000, sample_rate=16_000)

    assert chunk.channels == 1
    assert chunk.duration_s == 1.0


def test_stereo_duration() -> None:
    assert AudioChunk(pcm=b"\0" * 4 * 24_000, sample_rate=24_000, channels=2).duration_s == 1.0


@pytest.mark.parametrize(
    ("pcm", "rate", "channels"),
    [
        (b"\0\0", 0, 1),
        (b"\0\0", -16_000, 1),
        (b"\0\0", 7_999, 1),
        (b"\0\0", 192_001, 1),
        (b"\0", 16_000, 1),
        (b"\0\0", 16_000, 2),
        (b"\0\0", 16_000, 3),
        (b"\0\0", 16_000, 0),
    ],
)
def test_invalid_audio_chunk(pcm: bytes, rate: int, channels: int) -> None:
    with pytest.raises(InvalidAudio):
        AudioChunk(pcm=pcm, sample_rate=rate, channels=channels)


def test_audio_must_be_bytes() -> None:
    with pytest.raises(InvalidAudio):
        AudioChunk(pcm="audio", sample_rate=16_000)  # type: ignore[arg-type]


def test_empty_audio_is_allowed() -> None:
    assert AudioChunk(pcm=b"", sample_rate=16_000).duration_s == 0.0


def test_invalid_audio_is_a_domain_value_error() -> None:
    assert issubclass(InvalidAudio, DomainError)
    assert issubclass(InvalidAudio, ValueError)


@given(st.integers(min_value=0, max_value=4_000), st.sampled_from([8_000, 16_000, 24_000, 48_000]))
def test_any_whole_number_of_samples_is_valid(samples: int, rate: int) -> None:
    assert AudioChunk(pcm=b"\0\0" * samples, sample_rate=rate).duration_s == samples / rate


def test_transcript() -> None:
    transcript = Transcript(text="hello", language=EN, audio_s=1.5, processing_s=0.2)

    assert transcript.text == "hello"
    assert transcript.real_time_factor == pytest.approx(0.2 / 1.5)


def test_transcript_of_silence_has_no_real_time_factor() -> None:
    assert Transcript(text="", language=EN, audio_s=0.0, processing_s=0.1).real_time_factor is None


@pytest.mark.parametrize(("audio_s", "processing_s"), [(-1.0, 0.1), (1.0, -0.1)])
def test_transcript_timings_are_non_negative(audio_s: float, processing_s: float) -> None:
    with pytest.raises(InvalidAudio):
        Transcript(text="x", language=EN, audio_s=audio_s, processing_s=processing_s)


def test_transcript_text_must_be_text() -> None:
    with pytest.raises(InvalidAudio):
        Transcript(text=None, language=EN, audio_s=1.0, processing_s=0.1)  # type: ignore[arg-type]
