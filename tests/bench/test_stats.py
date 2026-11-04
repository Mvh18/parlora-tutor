"""Percentile summaries, the cold/warm split and the real-time factor."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.bench.common.stats import ColdWarm, Summary, percentile, real_time_factor, summarise

pytestmark = pytest.mark.unit

finite = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)


def test_summary_of_one_to_ten_follows_linear_interpolation() -> None:
    summary = summarise([float(v) for v in range(1, 11)])

    assert (summary.count, summary.min, summary.p50, summary.max, summary.mean) == (
        10,
        1.0,
        5.5,
        10.0,
        5.5,
    )
    assert summary.p95 == pytest.approx(9.55)


def test_order_of_values_does_not_matter() -> None:
    assert summarise([3.0, 1.0, 2.0]) == summarise([1.0, 2.0, 3.0])


def test_single_value_is_every_percentile() -> None:
    assert summarise([0.42]) == Summary(count=1, min=0.42, p50=0.42, p95=0.42, max=0.42, mean=0.42)


@pytest.mark.parametrize(("q", "expected"), [(0, 1.0), (100, 4.0), (50, 2.5), (25, 1.75)])
def test_percentile_endpoints_and_midpoints(q: float, expected: float) -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], q) == pytest.approx(expected)


@pytest.mark.parametrize("q", [-1, 100.5, math.nan])
def test_percentile_outside_zero_to_hundred_is_rejected(q: float) -> None:
    with pytest.raises(ValueError, match="percentile"):
        percentile([1.0], q)


def test_empty_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="no values"):
        summarise([])


def test_non_finite_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        summarise([1.0, math.inf])


@given(st.lists(finite, min_size=1, max_size=50))
def test_summary_is_ordered(values: list[float]) -> None:
    s = summarise(values)
    assert s.min <= s.p50 <= s.p95 <= s.max
    assert s.min <= s.mean * (1 + 1e-12) + 1e-9
    assert s.mean <= s.max * (1 + 1e-12) + 1e-9
    assert s.count == len(values)


def test_cold_run_is_kept_out_of_the_warm_summary() -> None:
    runs = ColdWarm(cold=9.0, warm=[0.5] * 10)

    assert runs.summary().p50 == 0.5
    assert runs.summary().max == 0.5
    assert runs.to_dict() == {
        "cold": 9.0,
        "warm": [0.5] * 10,
        "summary": {"count": 10, "min": 0.5, "p50": 0.5, "p95": 0.5, "max": 0.5, "mean": 0.5},
    }


def test_cold_run_may_be_missing() -> None:
    assert ColdWarm(cold=None, warm=[1.0]).to_dict()["cold"] is None


def test_warm_runs_are_required() -> None:
    with pytest.raises(ValueError, match="warm"):
        ColdWarm(cold=1.0, warm=[])


def test_real_time_factor() -> None:
    assert real_time_factor(processing_s=0.7, audio_s=10.0) == pytest.approx(0.07)


@pytest.mark.parametrize(("processing", "audio"), [(1.0, 0.0), (1.0, -2.0), (-0.1, 1.0)])
def test_real_time_factor_needs_positive_audio_and_non_negative_time(
    processing: float, audio: float
) -> None:
    with pytest.raises(ValueError, match="real-time factor"):
        real_time_factor(processing_s=processing, audio_s=audio)
