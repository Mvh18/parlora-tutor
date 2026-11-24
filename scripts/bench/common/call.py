"""Pure logic of the smoke-call benchmark (`scripts.bench.call`): standard library only.

The smoke call plays each synthetic fixture into a real call as one learner turn.
On the client side, `BotAudioTracker` watches the tutor's audio as it arrives:
a received frame is bot speech when its RMS is above a threshold, which tells
whether the turn got an answer, how much audio it got and when the first audio
arrived after the learner fell silent (a client-side cross-check of the server's
voice-to-voice latency). The server's own numbers, one record per turn and the
call summary, come from its turn observer; the checks hold the call against
the Phase 1 target over the turns Smart Turn released.
"""

import math
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from scripts.bench.common.budgets import BUDGETS, Check, evaluate_budget
from scripts.bench.common.stats import summarise

# Opus decodes digital silence to a few units of noise; speech is far above this.
DEFAULT_THRESHOLD = 200.0


def pcm16_rms(data: bytes) -> float:
    """Root mean square of native-endian signed 16-bit samples; 0.0 for no samples."""
    samples = array("h")
    samples.frombytes(data)
    if not samples:
        return 0.0
    return math.sqrt(math.fsum(s * s for s in samples) / len(samples))


class BotAudioTracker:
    """Tracks the loud (speech) frames of the tutor's audio since the current turn began."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._turn_at: float | None = None
        self.first_loud_at: float | None = None
        self.last_loud_at: float | None = None
        self.loud_s = 0.0

    def start_turn(self, at: float) -> None:
        self._turn_at = at
        self.first_loud_at = None
        self.last_loud_at = None
        self.loud_s = 0.0

    def frame(self, data: bytes, *, duration_s: float, at: float) -> None:
        if self._turn_at is None or at < self._turn_at or pcm16_rms(data) <= self.threshold:
            return
        if self.first_loud_at is None:
            self.first_loud_at = at
        self.last_loud_at = at
        self.loud_s += duration_s

    def quiet_for(self, at: float) -> float | None:
        """Seconds since the last loud frame; None when the turn has had no bot audio yet."""
        return None if self.last_loud_at is None else at - self.last_loud_at


@dataclass(frozen=True)
class ClientTurn:
    """One fixture played as a learner turn, as the client saw it. No text, ids only."""

    fixture: str
    utterance_s: float
    bot_audio_s: float
    first_bot_audio_s: float | None


def call_measurements(
    turns: Sequence[ClientTurn],
    server_turns: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    first = [t.first_bot_audio_s for t in turns if t.first_bot_audio_s is not None]
    return {
        "client_turns": [asdict(t) for t in turns],
        "answered": sum(t.bot_audio_s > 0 for t in turns),
        "client_first_audio_s": summarise(first).to_dict() if first else None,
        "server_turns": [dict(t) for t in server_turns],
        "summary": dict(summary),
    }


def _latency_check(name: str, value: float | None) -> Check:
    budget = BUDGETS[name]
    if value is None:
        return Check(
            name=name,
            status="fail",
            value=0.0,
            limit=budget.limit,
            unit=budget.unit,
            detail="no turn was released by Smart Turn; " + budget.source,
        )
    return evaluate_budget(budget, value)


def call_checks(turns: Sequence[ClientTurn], summary: Mapping[str, Any]) -> list[Check]:
    answered = sum(t.bot_audio_s > 0 for t in turns)
    return [
        Check(
            name="call_bot_audio_every_turn",
            status="pass" if turns and answered == len(turns) else "fail",
            value=answered,
            limit=len(turns),
            unit="turns",
            detail="every fixture played as a learner turn gets bot audio",
        ),
        _latency_check("call_model_p50", summary.get("model_p50_s")),
        _latency_check("call_model_p95", summary.get("model_p95_s")),
    ]
