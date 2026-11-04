"""Budgets and the VRAM rule, in one place.

Budgets are estimates: exceeding one is a `warn` and never fails the run. The one
hard rule, from the roadmap ("everything on the GPU fits in 8 GB"), is VRAM:
speech-to-text and the LLM loaded together, minus the baseline measured before
loading, must stay at or under `PARLORA_BENCH_VRAM_LIMIT_MB` (default 7680 MB, which
leaves 512 MB of the RTX 4060's 8 GB for the desktop and the driver). Exceeding it
is a `fail` and the benchmark exits with code 1.
"""

import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["pass", "warn", "fail"]
_STATUSES = ("pass", "warn", "fail")

VRAM_LIMIT_ENV = "PARLORA_BENCH_VRAM_LIMIT_MB"
DEFAULT_VRAM_LIMIT_MB = 7680


@dataclass(frozen=True)
class Budget:
    name: str
    limit: float
    unit: str
    source: str


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    value: float
    limit: float
    unit: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in _STATUSES:
            raise ValueError(f"status must be one of {', '.join(_STATUSES)}, got {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


BUDGETS = {
    budget.name: budget
    for budget in (
        # Voice-to-voice target is < 800 ms excellent, 800-1200 ms realistic (roadmap);
        # a 15 s utterance is the longest expected turn, ADR 0002 measured 0.8 s for 11 s.
        Budget(
            "stt_warm_p95",
            1.0,
            "s",
            "roadmap latency target; ADR 0002 measured 0.75-0.80 s for 11.35 s",
        ),
        # The LLM's share of an 800-1200 ms voice-to-voice turn.
        Budget("llm_warm_ttft", 0.4, "s", "roadmap latency target, LLM share of a turn"),
        # The first sentence of a reply must start playing quickly (roadmap).
        Budget("tts_first_audio", 0.5, "s", "roadmap latency target, TTS share of a turn"),
        # Synthesis must keep up with playback; ADR 0004 measured RTF 0.57 on CPU.
        Budget("tts_rtf", 1.0, "x", "real time; ADR 0004 measured 0.57 on CPU"),
    )
}


def evaluate_budget(budget: Budget, value: float) -> Check:
    """`pass` at or under the budget, `warn` above it."""
    return Check(
        name=budget.name,
        status="pass" if value <= budget.limit else "warn",
        value=value,
        limit=budget.limit,
        unit=budget.unit,
        detail=budget.source,
    )


def vram_limit_mb(env: Mapping[str, str] = os.environ) -> int:
    raw = env.get(VRAM_LIMIT_ENV)
    if raw is None:
        return DEFAULT_VRAM_LIMIT_MB
    try:
        limit = int(raw.strip())
    except ValueError:
        limit = 0
    if limit <= 0:
        raise ValueError(f"{VRAM_LIMIT_ENV} must be a positive number of MB, got {raw!r}")
    return limit


def evaluate_vram(model: str, *, used_mb: float, baseline_mb: float, limit_mb: int) -> Check:
    """`fail` when speech-to-text plus `model` use more than `limit_mb` above the baseline."""
    if baseline_mb > used_mb:
        raise ValueError(f"baseline {baseline_mb} MB is above the used VRAM {used_mb} MB")
    value = used_mb - baseline_mb
    return Check(
        name=f"vram_stt_plus_{model}",
        status="pass" if value <= limit_mb else "fail",
        value=value,
        limit=limit_mb,
        unit="MB",
        detail=f"used {used_mb} MB minus baseline {baseline_mb} MB; hard limit {VRAM_LIMIT_ENV}",
    )


def exit_code(checks: Iterable[Check]) -> int:
    """1 when any check failed; warnings keep the exit code at 0."""
    return 1 if any(check.status == "fail" for check in checks) else 0
