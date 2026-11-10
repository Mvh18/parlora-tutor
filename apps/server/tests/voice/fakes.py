"""Fake domain adapters for the voice tests: they record calls and answer at once."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from parlora_core.audio import AudioChunk, Transcript
from parlora_core.errors import DomainError
from parlora_core.languages import LanguageCode
from parlora_core.ports import Message

SPEECH = AudioChunk(b"\x01\x00" * 2_400, 24_000)  # 0.1 s of quiet audio


@dataclass
class FakeSpeechToText:
    text: str = "I went to the market yesterday."
    error: DomainError | None = None
    calls: list[AudioChunk] = field(default_factory=list)

    async def transcribe(self, audio: AudioChunk, *, language: LanguageCode) -> Transcript:
        self.calls.append(audio)
        if self.error is not None:
            raise self.error
        return Transcript(self.text, language, audio.duration_s, 0.01)


@dataclass
class FakeLanguageModel:
    chunks: Sequence[str] = ("Nice!", " What", " did", " you", " buy?")
    error: DomainError | None = None
    calls: list[list[Message]] = field(default_factory=list)

    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


@dataclass
class FakeTextToSpeech:
    audio: AudioChunk = SPEECH
    error: DomainError | None = None
    texts: list[str] = field(default_factory=list)

    async def stream_speech(
        self, text: str, *, language: LanguageCode, voice: str
    ) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        if self.error is not None:
            raise self.error
        yield self.audio
