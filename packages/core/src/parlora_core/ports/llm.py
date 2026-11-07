"""Language-model port."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of a conversation sent to the language model."""

    role: Role
    content: str


class LanguageModel(Protocol):
    """Writes the tutor's replies."""

    def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Yield the next assistant message for the conversation so far, piece by piece."""
        ...
