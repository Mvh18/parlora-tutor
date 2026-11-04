"""Budgets warn; the VRAM limit fails the run."""

import pytest

from scripts.bench.common.budgets import (
    BUDGETS,
    DEFAULT_VRAM_LIMIT_MB,
    VRAM_LIMIT_ENV,
    Check,
    evaluate_budget,
    evaluate_vram,
    exit_code,
    vram_limit_mb,
)

pytestmark = pytest.mark.unit


def test_documented_budgets() -> None:
    assert {name: b.limit for name, b in BUDGETS.items()} == {
        "stt_warm_p95": 1.0,
        "llm_warm_ttft": 0.4,
        "tts_first_audio": 0.5,
        "tts_rtf": 1.0,
    }
    assert all(b.source for b in BUDGETS.values())


def test_budget_exceeded_is_a_warning_and_exit_stays_zero() -> None:
    check = evaluate_budget(BUDGETS["stt_warm_p95"], 1.4)

    assert check.status == "warn"
    assert exit_code([check]) == 0


@pytest.mark.parametrize("value", [0.2, 1.0])
def test_budget_met_passes(value: float) -> None:
    assert evaluate_budget(BUDGETS["stt_warm_p95"], value).status == "pass"


def test_check_serialises_with_its_unit_and_source() -> None:
    check = evaluate_budget(BUDGETS["llm_warm_ttft"], 0.31)

    assert check.to_dict() == {
        "name": "llm_warm_ttft",
        "status": "pass",
        "value": 0.31,
        "limit": 0.4,
        "unit": "s",
        "detail": BUDGETS["llm_warm_ttft"].source,
    }


def test_vram_limit_exceeded_fails_with_a_non_zero_exit() -> None:
    check = evaluate_vram("qwen3:8b", used_mb=7_900, baseline_mb=0, limit_mb=7_680)

    assert check.name == "vram_stt_plus_qwen3:8b"
    assert check.status == "fail"
    assert exit_code([evaluate_budget(BUDGETS["tts_rtf"], 0.5), check]) == 1


def test_vram_baseline_is_subtracted() -> None:
    check = evaluate_vram("qwen3:4b", used_mb=8_000, baseline_mb=500, limit_mb=7_680)

    assert check.status == "pass"
    assert check.value == 7_500
    assert "baseline 500 MB" in check.detail


def test_vram_at_the_limit_passes() -> None:
    assert evaluate_vram("m", used_mb=7_680, baseline_mb=0, limit_mb=7_680).status == "pass"


def test_vram_baseline_above_used_is_rejected() -> None:
    with pytest.raises(ValueError, match="baseline"):
        evaluate_vram("m", used_mb=100, baseline_mb=200, limit_mb=7_680)


def test_no_checks_exit_zero() -> None:
    assert exit_code([]) == 0


def test_vram_limit_defaults_to_7680() -> None:
    assert vram_limit_mb({}) == DEFAULT_VRAM_LIMIT_MB == 7_680


def test_vram_limit_from_the_environment() -> None:
    assert vram_limit_mb({VRAM_LIMIT_ENV: " 6000 "}) == 6_000
    assert VRAM_LIMIT_ENV == "PARLORA_BENCH_VRAM_LIMIT_MB"


@pytest.mark.parametrize("raw", ["", "lots", "0", "-5", "7.5"])
def test_invalid_vram_limit_is_rejected(raw: str) -> None:
    with pytest.raises(ValueError, match=VRAM_LIMIT_ENV):
        vram_limit_mb({VRAM_LIMIT_ENV: raw})


def test_check_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="status"):
        Check(name="x", status="maybe", value=1, limit=1, unit="s", detail="")  # type: ignore[arg-type]
