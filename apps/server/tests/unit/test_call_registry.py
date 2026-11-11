"""The call registry: a call limit, a maximum duration and an idle timeout (design D6).

A fake clock drives the limits; nothing here opens a peer connection.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

import pytest

from parlora_server.voice.registry import (
    CallLimitError,
    CallRegistry,
    IceCandidate,
    UnknownCallError,
)

pytestmark = pytest.mark.unit


@dataclass
class FakeClock:
    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class FakeSession:
    closed: int = 0
    candidates: list[IceCandidate] = field(default_factory=list)

    async def add_ice_candidates(self, candidates: Sequence[IceCandidate]) -> None:
        self.candidates.extend(candidates)

    async def close(self) -> None:
        self.closed += 1


def registry(clock: FakeClock, *, max_calls: int = 1) -> CallRegistry:
    return CallRegistry(max_calls=max_calls, max_call_seconds=60, idle_seconds=20, clock=clock)


def starter(session: FakeSession) -> Callable[[str], Awaitable[tuple[FakeSession, str]]]:
    async def start(call_id: str) -> tuple[FakeSession, str]:
        return session, f"answer for {call_id}"

    return start


def test_a_call_gets_an_id_and_the_starter_answer() -> None:
    calls = registry(FakeClock())

    call_id, answer = asyncio.run(calls.open(starter(FakeSession())))

    assert answer == f"answer for {call_id}"
    assert calls.active == 1


def test_the_limit_refuses_a_call_without_starting_it() -> None:
    calls = registry(FakeClock(), max_calls=1)
    started: list[str] = []

    async def start(call_id: str) -> tuple[FakeSession, None]:
        started.append(call_id)
        return FakeSession(), None

    async def scenario() -> None:
        await calls.open(start)
        with pytest.raises(CallLimitError) as error:
            await calls.open(start)
        assert error.value.max_calls == 1

    asyncio.run(scenario())
    assert len(started) == 1


def test_a_pending_call_counts_against_the_limit() -> None:
    calls = registry(FakeClock(), max_calls=1)

    async def scenario() -> None:
        gate = asyncio.Event()

        async def slow(call_id: str) -> tuple[FakeSession, None]:
            await gate.wait()
            return FakeSession(), None

        first = asyncio.create_task(calls.open(slow))
        await asyncio.sleep(0)
        with pytest.raises(CallLimitError):
            await calls.open(starter(FakeSession()))
        gate.set()
        await first

    asyncio.run(scenario())
    assert calls.active == 1


def test_a_failed_start_frees_its_slot() -> None:
    calls = registry(FakeClock())

    async def broken(call_id: str) -> tuple[FakeSession, None]:
        raise ValueError("bad offer")

    async def scenario() -> None:
        with pytest.raises(ValueError, match="bad offer"):
            await calls.open(broken)
        await calls.open(starter(FakeSession()))

    asyncio.run(scenario())
    assert calls.active == 1


def test_ending_a_call_closes_it_and_frees_the_slot() -> None:
    calls = registry(FakeClock())
    session = FakeSession()

    async def scenario() -> None:
        call_id, _ = await calls.open(starter(session))
        await calls.end(call_id)
        await calls.open(starter(FakeSession()))

    asyncio.run(scenario())
    assert session.closed == 1


def test_unknown_calls_raise() -> None:
    calls = registry(FakeClock())

    async def scenario() -> None:
        with pytest.raises(UnknownCallError):
            await calls.end("nope")
        with pytest.raises(UnknownCallError):
            await calls.add_ice_candidates("nope", [])

    asyncio.run(scenario())


def test_ice_candidates_reach_the_session_and_count_as_activity() -> None:
    clock = FakeClock()
    calls = registry(clock)
    session = FakeSession()
    candidate = IceCandidate("candidate:1 1 udp 1 127.0.0.1 5000 typ host", "0", 0)

    async def scenario() -> None:
        call_id, _ = await calls.open(starter(session))
        clock.advance(15)
        await calls.add_ice_candidates(call_id, [candidate])
        clock.advance(15)
        assert await calls.sweep() == []

    asyncio.run(scenario())
    assert session.candidates == [candidate]


def test_a_call_ended_by_the_peer_frees_the_slot() -> None:
    calls = registry(FakeClock())
    session = FakeSession()

    async def scenario() -> None:
        call_id, _ = await calls.open(starter(session))
        calls.discard(call_id)
        calls.discard(call_id)  # a second notice is harmless
        await calls.open(starter(FakeSession()))

    asyncio.run(scenario())
    assert session.closed == 0, "the peer already closed it"


def test_the_maximum_duration_closes_a_busy_call() -> None:
    clock = FakeClock()
    calls = registry(clock)
    session = FakeSession()

    async def scenario() -> list[tuple[str, str]]:
        call_id, _ = await calls.open(starter(session))
        for _ in range(6):  # busy all along: never idle
            clock.advance(10)
            calls.touch(call_id)
        closed = await calls.sweep()
        await calls.open(starter(FakeSession()))
        return [(c, reason) for c, reason in closed if c == call_id]

    closed = asyncio.run(scenario())
    assert [reason for _, reason in closed] == ["max-duration"]
    assert session.closed == 1


def test_the_idle_timeout_closes_a_silent_call() -> None:
    clock = FakeClock()
    calls = registry(clock)
    session = FakeSession()

    async def scenario() -> list[tuple[str, str]]:
        call_id, _ = await calls.open(starter(session))
        clock.advance(19)
        assert await calls.sweep() == []
        calls.touch(call_id)
        clock.advance(19)
        assert await calls.sweep() == []
        clock.advance(2)
        return await calls.sweep()

    closed = asyncio.run(scenario())
    assert [reason for _, reason in closed] == ["idle"]
    assert session.closed == 1
    assert calls.active == 0


def test_touching_an_unknown_call_is_ignored() -> None:
    registry(FakeClock()).touch("gone")


def test_a_session_that_fails_to_close_still_frees_the_slot() -> None:
    clock = FakeClock()
    calls = registry(clock)

    class Stuck(FakeSession):
        async def close(self) -> None:
            raise RuntimeError("peer connection already gone")

    async def scenario() -> None:
        await calls.open(starter(Stuck()))
        clock.advance(61)
        await calls.sweep()

    asyncio.run(scenario())
    assert calls.active == 0


def test_close_all_ends_every_call() -> None:
    calls = registry(FakeClock(), max_calls=2)
    sessions = [FakeSession(), FakeSession()]

    async def scenario() -> None:
        for session in sessions:
            await calls.open(starter(session))
        await calls.close_all()

    asyncio.run(scenario())
    assert [s.closed for s in sessions] == [1, 1]
    assert calls.active == 0


def test_the_watchdog_sweeps_until_cancelled() -> None:
    clock = FakeClock()
    calls = registry(clock)
    session = FakeSession()

    async def scenario() -> None:
        await calls.open(starter(session))
        clock.advance(61)
        watchdog = asyncio.create_task(calls.run_watchdog(interval=0.001))
        for _ in range(100):
            if session.closed:
                break
            await asyncio.sleep(0.001)
        watchdog.cancel()
        with pytest.raises(asyncio.CancelledError):
            await watchdog

    asyncio.run(scenario())
    assert session.closed == 1


@pytest.mark.parametrize(
    ("max_calls", "max_call_seconds", "idle_seconds"), [(0, 60, 20), (1, 0, 20), (1, 60, 0)]
)
def test_limits_must_be_positive(
    max_calls: int, max_call_seconds: float, idle_seconds: float
) -> None:
    with pytest.raises(ValueError, match="positive"):
        CallRegistry(
            max_calls=max_calls, max_call_seconds=max_call_seconds, idle_seconds=idle_seconds
        )
