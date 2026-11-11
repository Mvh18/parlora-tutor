"""The calls the server holds open, and the limits on them (design D6).

At most `max_calls` calls run at once (one by default: one GPU serves one
learner); a call that is still being set up counts against the limit. A call ends
when the client ends it, when its peer connection closes, when it reaches
`max_call_seconds`, or when nobody has spoken for `idle_seconds`. The registry
knows nothing about WebRTC: a starter sets the call up and hands back a session
the registry can feed ICE candidates to and close.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from parlora_core.session import SessionSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IceCandidate:
    """A remote ICE candidate; an empty `candidate` marks the end of candidates."""

    candidate: str
    sdp_mid: str | None
    sdp_mline_index: int | None


@dataclass(frozen=True)
class SdpDescription:
    sdp: str
    type: str


class CallSession(Protocol):
    async def add_ice_candidates(self, candidates: Sequence[IceCandidate]) -> None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class CallHooks:
    """What a running call reports back: someone spoke, or the call ended on its own."""

    activity: Callable[[], None]
    ended: Callable[[], None]


class CallStarter(Protocol):
    """Sets a call up from the client's offer and answers it (the WebRTC side)."""

    async def __call__(
        self, offer: SdpDescription, settings: SessionSettings, hooks: CallHooks
    ) -> tuple[CallSession, SdpDescription]: ...


class InvalidOfferError(ValueError):
    """The peer connection could not use the client's offer."""


class CallLimitError(Exception):
    def __init__(self, max_calls: int) -> None:
        super().__init__(f"{max_calls} call(s) already active")
        self.max_calls = max_calls


class UnknownCallError(LookupError):
    """No active call has this id."""


@dataclass
class _Call:
    session: CallSession
    started: float
    last_activity: float


class CallRegistry:
    def __init__(
        self,
        *,
        max_calls: int,
        max_call_seconds: float,
        idle_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_calls < 1 or max_call_seconds <= 0 or idle_seconds <= 0:
            raise ValueError("call limits must be positive")
        self.max_calls = max_calls
        self.max_call_seconds = max_call_seconds
        self.idle_seconds = idle_seconds
        self._clock = clock
        self._calls: dict[str, _Call] = {}
        self._pending = 0

    @property
    def active(self) -> int:
        return len(self._calls)

    async def open[T](
        self, start: Callable[[str], Awaitable[tuple[CallSession, T]]]
    ) -> tuple[str, T]:
        """Reserve a slot, set the call up with `start(call_id)` and register it."""
        # No await between the check and the reservation: the event loop cannot
        # interleave another offer here.
        if len(self._calls) + self._pending >= self.max_calls:
            raise CallLimitError(self.max_calls)
        self._pending += 1
        call_id = uuid.uuid4().hex
        try:
            session, answer = await start(call_id)
        finally:
            self._pending -= 1
        now = self._clock()
        self._calls[call_id] = _Call(session, started=now, last_activity=now)
        logger.info("call %s started (%d active)", call_id, len(self._calls))
        return call_id, answer

    def touch(self, call_id: str) -> None:
        """Someone spoke (or signalled) in the call: it is not idle."""
        call = self._calls.get(call_id)
        if call is not None:
            call.last_activity = self._clock()

    async def add_ice_candidates(self, call_id: str, candidates: Sequence[IceCandidate]) -> None:
        call = self._get(call_id)
        call.last_activity = self._clock()
        await call.session.add_ice_candidates(candidates)

    async def end(self, call_id: str, reason: str = "ended by the client") -> None:
        self._get(call_id)
        await self._close(call_id, reason)

    def discard(self, call_id: str) -> None:
        """The call ended on its own (the peer left, the pipeline stopped)."""
        if self._calls.pop(call_id, None) is not None:
            logger.info("call %s ended (%d active)", call_id, len(self._calls))

    async def sweep(self) -> list[tuple[str, str]]:
        """Close the calls past their maximum duration or idle timeout."""
        now = self._clock()
        expired = [
            (call_id, "max-duration" if now - call.started >= self.max_call_seconds else "idle")
            for call_id, call in list(self._calls.items())
            if now - call.started >= self.max_call_seconds
            or now - call.last_activity >= self.idle_seconds
        ]
        for call_id, reason in expired:
            await self._close(call_id, reason)
        return expired

    async def run_watchdog(self, interval: float = 1.0) -> None:
        while True:
            await asyncio.sleep(interval)
            await self.sweep()

    async def close_all(self) -> None:
        for call_id in list(self._calls):
            await self._close(call_id, "server shutting down")

    def _get(self, call_id: str) -> _Call:
        try:
            return self._calls[call_id]
        except KeyError:
            raise UnknownCallError(call_id) from None

    async def _close(self, call_id: str, reason: str) -> None:
        call = self._calls.pop(call_id, None)
        if call is None:
            return
        logger.info("call %s ended: %s (%d active)", call_id, reason, len(self._calls))
        try:
            await call.session.close()
        except Exception:
            logger.warning("call %s did not close cleanly", call_id, exc_info=True)
