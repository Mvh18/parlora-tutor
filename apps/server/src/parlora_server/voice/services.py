"""Thin Pipecat services over the domain ports (design D1).

Each service only translates frames to a port call and back: Pipecat runs the
frame flow, interruptions and metrics, while the models stay behind
`parlora_core.ports` and their server adapters. A Pipecat upgrade touches this
module and the pipeline builder only.

- `SpeechToTextService`: one learner turn (the audio between VAD start and stop)
  becomes one `SpeechToText.transcribe` call and one `TranscriptionFrame`.
- `LanguageModelService`: each context frame becomes one `LanguageModel.stream_reply`
  call; every streamed chunk is pushed as it arrives.
- `TextToSpeechService`: the streamed reply is cut into clauses
  (`ClauseTextAggregator`), cleaned of markdown and emojis, and each clause becomes
  one `TextToSpeech.stream_speech` call whose audio is resampled to the output rate.

Model failures (`DomainError`) become Pipecat error frames; transcripts and replies
are never logged here.
"""

import dataclasses
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import Any

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import is_given

from parlora_core.audio import AudioChunk
from parlora_core.errors import DomainError
from parlora_core.languages import LanguageCode
from parlora_core.ports import LanguageModel, Message, SpeechToText, TextToSpeech
from parlora_core.speech import spoken_text
from parlora_server.voice.aggregator import ClauseTextAggregator


def complete[S](settings: S) -> S:
    """Pipecat settings with every field the port does not use set to None.

    Pipecat wants every settings field initialised; the knobs live in our adapters.
    """
    for field in dataclasses.fields(settings):  # type: ignore[arg-type]
        if not field.name.startswith("_") and not is_given(getattr(settings, field.name)):
            setattr(settings, field.name, None)
    return settings


_ROLES = {"system": "system", "developer": "system", "user": "user", "assistant": "assistant"}


class SpeechToTextService(SegmentedSTTService):
    """Transcribes each learner turn through the `SpeechToText` port."""

    def __init__(self, stt: SpeechToText, *, language: LanguageCode, **kwargs: Any) -> None:
        super().__init__(settings=complete(STTSettings()), **kwargs)
        self._stt = stt
        self._language = language

    @property
    def wants_wav_segments(self) -> bool:
        """Receive raw 16-bit PCM: the port takes PCM, not a WAV container."""
        return False

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        await self.start_processing_metrics()
        try:
            transcript = await self._stt.transcribe(
                AudioChunk(audio, self.sample_rate), language=self._language
            )
        except DomainError as error:
            yield ErrorFrame(error=f"speech-to-text failed: {type(error).__name__}")
            return
        finally:
            await self.stop_processing_metrics()
        if transcript.text:
            yield TranscriptionFrame(transcript.text, self._user_id, time_now_iso8601())


def _text(content: Any) -> str | None:
    """The text of an OpenAI-style message content: a string or a list of text parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, Iterable):
        parts = [
            str(part.get("text", ""))
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        return "".join(parts) if parts else None
    return None


def to_messages(context: LLMContext) -> list[Message]:
    """The conversation in a Pipecat context as domain messages (text only)."""
    messages = []
    for entry in context.get_messages():
        if not isinstance(entry, Mapping):
            continue
        role = _ROLES.get(str(entry.get("role")))
        text = _text(entry.get("content"))
        if role is not None and text:
            messages.append(Message(role, text))  # type: ignore[arg-type]
    return messages


class LanguageModelService(LLMService):
    """Streams the tutor's reply through the `LanguageModel` port."""

    def __init__(self, llm: LanguageModel, **kwargs: Any) -> None:
        super().__init__(settings=complete(LLMSettings()), **kwargs)
        self._llm = llm

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        await self.start_ttfb_metrics()
        try:
            async for text in self._llm.stream_reply(to_messages(frame.context)):
                await self.stop_ttfb_metrics()
                await self._push_llm_text(text)
        except DomainError as error:
            await self.push_error(error_msg=f"language model failed: {error}", exception=error)
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())


async def _spoken(text: str, _type: Any) -> str:
    return spoken_text(text)


class TextToSpeechService(TTSService):
    """Speaks the reply clause by clause through the `TextToSpeech` port."""

    def __init__(
        self, tts: TextToSpeech, *, language: LanguageCode, voice: str, **kwargs: Any
    ) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            text_transforms=[("*", _spoken)],
            settings=complete(TTSSettings(voice=voice)),
            **kwargs,
        )
        # Pipecat builds its sentence aggregator in __init__ and takes no other;
        # swap in the clause rule (pinned Pipecat version, covered by tests).
        self._text_aggregator = ClauseTextAggregator()
        self._tts = tts
        self._language = language
        self._voice = voice

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if not text.strip():
            return
        await self.start_tts_usage_metrics(text)
        try:
            async for chunk in self._tts.stream_speech(
                text, language=self._language, voice=self._voice
            ):
                await self.stop_ttfb_metrics()
                audio = await self._resampler.resample(
                    chunk.pcm, chunk.sample_rate, self.sample_rate
                )
                yield TTSAudioRawFrame(
                    audio=audio,
                    sample_rate=self.sample_rate,
                    num_channels=chunk.channels,
                    context_id=context_id,
                )
        except DomainError as error:
            yield ErrorFrame(error=f"text-to-speech failed: {type(error).__name__}")
        finally:
            await self.stop_ttfb_metrics()
