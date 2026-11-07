"""Speech-to-text port."""

from typing import Protocol

from parlora_core.audio import AudioChunk, Transcript
from parlora_core.languages import LanguageCode


class SpeechToText(Protocol):
    """Transcribes one learner turn."""

    async def transcribe(self, audio: AudioChunk, *, language: LanguageCode) -> Transcript:
        """Return what was said in `audio`, spoken in `language`."""
        ...
