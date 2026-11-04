"""Summary statistics shared by every benchmark.

Percentile method: linear interpolation between the closest ranks, the default of
NumPy (`method="linear"`) and type 7 of Hyndman and Fan. For sorted values
x[0..n-1] the q-th percentile sits at rank h = (n - 1) * q / 100 and is
x[floor(h)] + (h - floor(h)) * (x[floor(h) + 1] - x[floor(h)]). For the values 1 to
10 this gives p50 = 5.5 and p95 = 9.55.

Every benchmark reports one cold run on its own and summarises only the warm runs.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field


def _check(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("no values to summarise")
    if not all(math.isfinite(v) for v in values):
        raise ValueError("values must be finite")
    return sorted(values)


def percentile(values: Sequence[float], q: float) -> float:
    """The q-th percentile (0 to 100) by linear interpolation between closest ranks."""
    if not 0 <= q <= 100:
        raise ValueError(f"percentile must be between 0 and 100, got {q}")
    ordered = _check(values)
    rank = (len(ordered) - 1) * q / 100
    low = math.floor(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


@dataclass(frozen=True)
class Summary:
    count: int
    min: float
    p50: float
    p95: float
    max: float
    mean: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarise(values: Sequence[float]) -> Summary:
    ordered = _check(values)
    return Summary(
        count=len(ordered),
        min=ordered[0],
        p50=percentile(ordered, 50),
        p95=percentile(ordered, 95),
        max=ordered[-1],
        mean=math.fsum(ordered) / len(ordered),
    )


@dataclass(frozen=True)
class ColdWarm:
    """One optional cold run and the warm runs; only the warm runs are summarised."""

    cold: float | None
    warm: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.warm:
            raise ValueError("at least one warm run is required")

    def summary(self) -> Summary:
        return summarise(self.warm)

    def to_dict(self) -> dict[str, object]:
        return {"cold": self.cold, "warm": list(self.warm), "summary": self.summary().to_dict()}


def real_time_factor(*, processing_s: float, audio_s: float) -> float:
    """Processing time divided by audio duration: below 1.0 is faster than real time."""
    if audio_s <= 0 or processing_s < 0:
        raise ValueError(
            "real-time factor needs a positive audio duration and a non-negative time, "
            f"got {processing_s} s for {audio_s} s"
        )
    return processing_s / audio_s
