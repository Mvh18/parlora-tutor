"""Text-to-speech port."""

from collections.abc import AsyncIterator
from typing import Protocol

from parlora_core.audio import AudioChunk
from parlora_core.languages import LanguageCode


class TextToSpeech(Protocol):
    """Speaks the tutor's replies."""

    def stream_speech(
        self, text: str, *, language: LanguageCode, voice: str
    ) -> AsyncIterator[AudioChunk]:
        """Yield audio for `text`, spoken in `language` with `voice`, as it is synthesised."""
        ...
