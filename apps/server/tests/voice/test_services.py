"""The Pipecat services over the domain ports, driven with Pipecat's test helpers."""

import asyncio
from collections.abc import Sequence

import pytest
from fakes import SPEECH, FakeLanguageModel, FakeSpeechToText, FakeTextToSpeech
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import run_test

from parlora_core.errors import LanguageModelError, UnsupportedLanguage
from parlora_core.languages import LanguageCode
from parlora_core.ports import Message
from parlora_server.voice import services

pytestmark = pytest.mark.unit

EN = LanguageCode("en")


def run(
    processor: FrameProcessor, frames: Sequence[Frame]
) -> tuple[Sequence[Frame], Sequence[Frame]]:
    return asyncio.run(run_test(processor, frames_to_send=frames))


def of_type[F: Frame](frames: Sequence[Frame], kind: type[F]) -> list[F]:
    return [f for f in frames if isinstance(f, kind)]


# ── language model ──────────────────────────────────────────────────────────


def context(*messages: dict[str, object]) -> LLMContext:
    return LLMContext(messages=list(messages))  # type: ignore[arg-type]


def test_context_messages_become_domain_messages() -> None:
    ctx = context(
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image_url"}]},
        {"role": "assistant", "content": "Hello!"},
        {"role": "tool", "content": "ignored"},
        {"role": "user", "content": ""},
        {"role": "user", "content": [{"type": "image_url"}]},
        {"role": "user", "content": 42},
    )

    assert services.to_messages(ctx) == [
        Message("system", "Be brief."),
        Message("user", "Hi"),
        Message("assistant", "Hello!"),
    ]


def test_each_streamed_chunk_is_pushed_between_start_and_end() -> None:
    llm = FakeLanguageModel()
    ctx = context({"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"})

    down, _ = run(services.LanguageModelService(llm), [LLMContextFrame(ctx)])

    reply = (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame)
    kinds = [type(f) for f in down if isinstance(f, reply)]
    texts = [f.text for f in of_type(down, LLMTextFrame)]
    assert kinds[0] is LLMFullResponseStartFrame
    assert kinds[-1] is LLMFullResponseEndFrame
    assert texts == ["Nice!", " What", " did", " you", " buy?"]
    assert llm.calls == [[Message("system", "Be brief."), Message("user", "Hi")]]


def test_a_language_model_error_is_pushed_upstream_and_the_reply_still_ends() -> None:
    llm = FakeLanguageModel(chunks=["Hi"], error=LanguageModelError("qwen: Ollama error: boom"))

    down, up = run(services.LanguageModelService(llm), [LLMContextFrame(context())])

    assert any("boom" in e.error for e in of_type(up, ErrorFrame))
    assert isinstance(
        [f for f in down if isinstance(f, (LLMTextFrame, LLMFullResponseEndFrame))][-1],
        LLMFullResponseEndFrame,
    )


# ── text-to-speech ──────────────────────────────────────────────────────────


def reply_frames(*tokens: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(t) for t in tokens),
        LLMFullResponseEndFrame(),
    ]


def test_clauses_reach_text_to_speech_before_the_reply_ends() -> None:
    tts = FakeTextToSpeech()
    tokens = ["Sure,", " I", " love", " that", " question.", " What", " do", " you", " think?"]

    down, _ = run(
        services.TextToSpeechService(tts, language=EN, voice="af_heart"), reply_frames(*tokens)
    )

    assert tts.texts == ["Sure, I love that question.", "What do you think?"]
    assert of_type(down, TTSStartedFrame)
    assert of_type(down, TTSStoppedFrame)


def test_markdown_and_emojis_are_not_spoken() -> None:
    tts = FakeTextToSpeech()

    run(
        services.TextToSpeechService(tts, language=EN, voice="af_heart"),
        reply_frames("**Great!**", " Let's", " talk", " 😀"),
    )

    assert tts.texts == ["Great! Let's talk"]


def test_audio_is_resampled_to_the_output_rate() -> None:
    tts = FakeTextToSpeech()

    down, _ = run(
        services.TextToSpeechService(tts, language=EN, voice="af_heart", sample_rate=16_000),
        reply_frames("Hello", " there."),
    )

    audio = of_type(down, TTSAudioRawFrame)
    assert audio
    assert {f.sample_rate for f in audio} == {16_000}
    assert sum(len(f.audio) for f in audio) < len(SPEECH.pcm)


def test_a_text_to_speech_error_becomes_an_error_frame() -> None:
    tts = FakeTextToSpeech(error=UnsupportedLanguage("Kokoro speaks only en"))

    down, up = run(
        services.TextToSpeechService(tts, language=EN, voice="af_heart"),
        reply_frames("Hello", " there."),
    )

    errors = of_type([*down, *up], ErrorFrame)
    assert any("UnsupportedLanguage" in e.error for e in errors)
    assert not of_type(down, TTSAudioRawFrame)


# ── speech-to-text ──────────────────────────────────────────────────────────


def test_one_turn_of_audio_becomes_one_transcription() -> None:
    stt = FakeSpeechToText()
    service = services.SpeechToTextService(stt, language=EN, sample_rate=16_000)

    async def scenario() -> list[str]:
        texts = []
        async for frame in service.run_stt(b"\x00\x00" * 16_000):
            assert isinstance(frame, TranscriptionFrame)
            texts.append(frame.text)
        return texts

    service._sample_rate = 16_000  # set by StartFrame in a running pipeline
    assert asyncio.run(scenario()) == ["I went to the market yesterday."]
    (audio,) = stt.calls
    assert (audio.sample_rate, audio.duration_s) == (16_000, 1.0)
    assert service.wants_wav_segments is False


def test_silence_transcribed_as_nothing_pushes_no_frame() -> None:
    service = services.SpeechToTextService(FakeSpeechToText(text=""), language=EN)
    service._sample_rate = 16_000

    async def scenario() -> list[Frame | None]:
        return [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    assert asyncio.run(scenario()) == []


def test_a_speech_to_text_error_becomes_an_error_frame() -> None:
    from parlora_core.errors import InvalidAudio

    service = services.SpeechToTextService(FakeSpeechToText(error=InvalidAudio("odd")), language=EN)
    service._sample_rate = 16_000

    async def scenario() -> list[Frame | None]:
        return [frame async for frame in service.run_stt(b"\x00\x00" * 160)]

    (frame,) = asyncio.run(scenario())
    assert isinstance(frame, ErrorFrame)
    assert "InvalidAudio" in frame.error
