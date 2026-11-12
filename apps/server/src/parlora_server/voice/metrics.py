"""Turn latency of a call, logged without its content (design D8).

`TurnMetricsObserver` watches a call's frames and, for every learner turn, logs
one INFO record with:

- `voice_to_voice_s`: from the end of the learner's speech to the first bot
  audio frame sent. The end of speech is the VAD's decision time minus its
  `stop_secs`, the silence it waited out (the same anchor as Pipecat's
  `UserBotLatencyObserver`); the first audio frame sent is the output
  transport's `BotStartedSpeakingFrame`.
- `stt_s`, `llm_ttft_s`, `tts_first_audio_s`: the first speech-to-text
  processing time, LLM time to first token and text-to-speech time to first
  audio Pipecat's metrics report in the turn.
- `interrupted`: the learner spoke before the reply ended (barge-in).

A turn ends when the bot stops speaking or when the learner interrupts the reply;
a reply interrupted before any audio has no voice-to-voice latency. If the output
runs dry between two clauses long enough for Pipecat to report the bot stopped,
the turn ends there: its latency is already measured, but a barge-in after that
gap is not flagged. When the call
ends, `summary()` gives the turn count and the p50 and p95 voice-to-voice latency.

What the learner said and the tutor answered is logged only at DEBUG. The record
carries the numbers both in its message and as `extra` attributes.
"""

import logging
import math
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

logger = logging.getLogger(__name__)

# The RTVI server message the call page reads the last turn's latency from.
LATENCY_MESSAGE_TYPE = "turn-latency"


@dataclass(frozen=True)
class TurnLatency:
    """The measured latencies of one turn, in seconds; None when not measured."""

    voice_to_voice_s: float | None
    stt_s: float | None
    llm_ttft_s: float | None
    tts_first_audio_s: float | None
    interrupted: bool

    def client_message(self) -> dict[str, Any]:
        """The server message sent to the client once the bot starts speaking."""
        return {"type": LATENCY_MESSAGE_TYPE, **_milliseconds(self)}


@dataclass(frozen=True)
class CallSummary:
    turns: int
    measured: int
    interrupted: int
    p50_s: float | None
    p95_s: float | None


def percentile(values: Sequence[float], q: float) -> float:
    """The `q`-th percentile (0 to 100) with linear interpolation between ranks."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def summarise(turns: Sequence[TurnLatency]) -> CallSummary:
    latencies = [t.voice_to_voice_s for t in turns if t.voice_to_voice_s is not None]
    return CallSummary(
        turns=len(turns),
        measured=len(latencies),
        interrupted=sum(t.interrupted for t in turns),
        p50_s=percentile(latencies, 50) if latencies else None,
        p95_s=percentile(latencies, 95) if latencies else None,
    )


def _milliseconds(turn: TurnLatency) -> dict[str, Any]:
    def ms(seconds: float | None) -> int | None:
        return None if seconds is None else round(seconds * 1000)

    return {
        "voice_to_voice_ms": ms(turn.voice_to_voice_s),
        "stt_ms": ms(turn.stt_s),
        "llm_ttft_ms": ms(turn.llm_ttft_s),
        "tts_first_audio_ms": ms(turn.tts_first_audio_s),
        "interrupted": turn.interrupted,
    }


def _fields(values: dict[str, Any]) -> str:
    return " ".join(f"{key}={'-' if value is None else value}" for key, value in values.items())


@dataclass
class _Turn:
    """A turn being measured: from the learner's end of speech to the reply's end."""

    user_stopped_at: float
    released: bool = False
    voice_to_voice_s: float | None = None
    stt_s: float | None = None
    llm_ttft_s: float | None = None
    tts_first_audio_s: float | None = None
    bot_speaking: bool = False
    said: list[str] = field(default_factory=list)
    answered: list[str] = field(default_factory=list)

    def result(self, *, interrupted: bool) -> TurnLatency:
        return TurnLatency(
            self.voice_to_voice_s,
            self.stt_s,
            self.llm_ttft_s,
            self.tts_first_audio_s,
            interrupted,
        )


LatencySink = Callable[[TurnLatency], Coroutine[Any, Any, None]]


class TurnMetricsObserver(BaseObserver):
    """Measures every turn of a call and logs it without its text."""

    def __init__(
        self,
        *,
        on_latency: LatencySink | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        super().__init__(observe_every_push=False)
        self._on_latency = on_latency
        self._clock = clock
        self._turn: _Turn | None = None
        self.turns: list[TurnLatency] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        # The output transport pushes the bot speaking frames both ways; one is enough.
        if data.direction != FrameDirection.DOWNSTREAM:
            return
        frame, turn = data.frame, self._turn
        if isinstance(frame, VADUserStartedSpeakingFrame):
            # The learner talks over the reply, or before its first audio; otherwise
            # they only paused inside the turn and it goes on.
            if turn is not None and (turn.bot_speaking or turn.released):
                self._finish(turn, interrupted=True)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._learner_stopped(turn, frame.timestamp - frame.stop_secs)
        elif turn is None:
            return
        elif isinstance(frame, UserStoppedSpeakingFrame):
            turn.released = True
        elif isinstance(frame, MetricsFrame):
            _take_metrics(turn, frame, data.source)
        elif isinstance(frame, BotStartedSpeakingFrame):
            await self._bot_started(turn)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            if turn.bot_speaking:
                self._finish(turn, interrupted=False)
        elif isinstance(frame, TranscriptionFrame):
            turn.said.append(frame.text)
        elif isinstance(frame, LLMTextFrame):
            turn.answered.append(frame.text)

    def summary(self) -> CallSummary:
        return summarise(self.turns)

    def log_summary(self) -> CallSummary:
        """Log the call summary at INFO and return it."""
        summary = self.summary()
        values = {
            "turns": summary.turns,
            "measured": summary.measured,
            "interrupted": summary.interrupted,
            "p50_ms": None if summary.p50_s is None else round(summary.p50_s * 1000),
            "p95_ms": None if summary.p95_s is None else round(summary.p95_s * 1000),
        }
        logger.info("call summary %s", _fields(values), extra={"call_summary": asdict(summary)})
        return summary

    def _learner_stopped(self, turn: _Turn | None, at: float) -> None:
        if turn is None:
            self._turn = _Turn(user_stopped_at=at)
        elif not turn.released:
            turn.user_stopped_at = at  # a pause inside the turn: measure from the last stop

    async def _bot_started(self, turn: _Turn) -> None:
        if turn.bot_speaking:
            return
        turn.bot_speaking = True
        turn.voice_to_voice_s = max(0.0, self._clock() - turn.user_stopped_at)
        if self._on_latency is not None:
            await self._on_latency(turn.result(interrupted=False))

    def _finish(self, turn: _Turn, *, interrupted: bool) -> None:
        self._turn = None
        result = turn.result(interrupted=interrupted)
        self.turns.append(result)
        values = _milliseconds(result)
        logger.info("turn %s", _fields(values), extra={"turn_latency": values})
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "turn text learner=%r tutor=%r", " ".join(turn.said), "".join(turn.answered)
            )


def _first_positive(current: float | None, value: float) -> float | None:
    """Keep the first measurement of the turn; a zero TTFB is a cancelled one."""
    return current if current is not None or value <= 0 else value


def _take_metrics(turn: _Turn, frame: MetricsFrame, source: Any) -> None:
    for data in frame.data:
        if isinstance(source, STTService) and isinstance(data, ProcessingMetricsData):
            turn.stt_s = _first_positive(turn.stt_s, data.value)
        elif isinstance(source, LLMService) and isinstance(data, TTFBMetricsData):
            turn.llm_ttft_s = _first_positive(turn.llm_ttft_s, data.value)
        elif isinstance(source, TTSService) and isinstance(data, TTFBMetricsData):
            turn.tts_first_audio_s = _first_positive(turn.tts_first_audio_s, data.value)
