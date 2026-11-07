"""Session repository port (its shape is settled with persistence in Phase 5)."""

from typing import Protocol

from parlora_core.learner import LearnerRef
from parlora_core.session import SessionSettings


class SessionRepository(Protocol):
    """Stores conversation sessions, keyed by an opaque session id."""

    async def add(self, session_id: str, settings: SessionSettings) -> None:
        """Store a new session."""

    async def get(self, session_id: str) -> SessionSettings | None:
        """Return the session's settings, or None when it does not exist."""

    async def list_for(self, learner_ref: LearnerRef) -> list[SessionSettings]:
        """Return the learner's sessions, newest first."""
