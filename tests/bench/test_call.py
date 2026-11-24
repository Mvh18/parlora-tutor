"""The smoke call's pure logic: loudness, client turns, measurements and checks."""

import math
from array import array

import pytest

from scripts.bench.common.call import (
    BotAudioTracker,
    ClientTurn,
    call_checks,
    call_measurements,
    pcm16_rms,
)

pytestmark = pytest.mark.unit


def pcm(*samples: int) -> bytes:
    return array("h", samples).tobytes()


def summary(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "turns": 3,
        "measured": 3,
        "interrupted": 0,
        "p50_s": 0.9,
        "p95_s": 1.4,
        "model_turns": 3,
        "model_p50_s": 0.9,
        "model_p95_s": 1.4,
        "fallback_turns": 0,
        "fallback_wait_p50_s": None,
    }
    values.update(overrides)
    return values


def turn(fixture: str = "a1-greeting", bot_audio_s: float = 2.0) -> ClientTurn:
    return ClientTurn(fixture, utterance_s=3.4, bot_audio_s=bot_audio_s, first_bot_audio_s=0.8)


def test_rms_of_silence_is_zero_and_of_a_square_wave_its_amplitude() -> None:
    assert pcm16_rms(b"") == 0.0
    assert pcm16_rms(pcm(0, 0, 0)) == 0.0
    assert pcm16_rms(pcm(1000, -1000, 1000, -1000)) == pytest.approx(1000.0)
    assert pcm16_rms(pcm(3, 4)) == pytest.approx(math.sqrt(12.5))


def test_the_tracker_measures_loud_audio_after_the_turn_mark() -> None:
    tracker = BotAudioTracker(threshold=100.0)
    tracker.frame(pcm(500, -500), duration_s=0.02, at=0.5)  # before the mark: ignored
    tracker.start_turn(at=1.0)
    tracker.frame(pcm(0, 0), duration_s=0.02, at=1.1)
    assert tracker.quiet_for(at=1.2) is None
    tracker.frame(pcm(500, -500), duration_s=0.02, at=1.9)
    tracker.frame(pcm(500, -500), duration_s=0.02, at=1.92)

    assert tracker.first_loud_at == 1.9
    assert tracker.last_loud_at == 1.92
    assert tracker.loud_s == pytest.approx(0.04)
    assert tracker.quiet_for(at=3.0) == pytest.approx(1.08)


def test_the_tracker_is_quiet_forever_before_any_bot_audio() -> None:
    tracker = BotAudioTracker(threshold=100.0)
    tracker.start_turn(at=1.0)

    assert tracker.quiet_for(at=5.0) is None


def test_measurements_carry_client_turns_server_turns_and_the_summary() -> None:
    turns = [turn("a1-greeting"), turn("b1-films", bot_audio_s=1.5)]
    server = [{"voice_to_voice_ms": 900, "end_by": "model"}]

    m = call_measurements(turns, server, summary())

    assert m["client_turns"][1] == {
        "fixture": "b1-films",
        "utterance_s": 3.4,
        "bot_audio_s": 1.5,
        "first_bot_audio_s": 0.8,
    }
    assert m["server_turns"] == server
    assert m["summary"]["model_p95_s"] == 1.4
    assert m["answered"] == 2
    assert m["client_first_audio_s"]["count"] == 2


def test_every_turn_answered_and_within_target_passes() -> None:
    checks = {c.name: c for c in call_checks([turn(), turn()], summary())}

    assert {c.status for c in checks.values()} == {"pass"}
    assert checks["call_bot_audio_every_turn"].value == 2
    assert checks["call_model_p50"].limit == 1.2
    assert checks["call_model_p95"].limit == 1.5


def test_a_turn_without_bot_audio_fails() -> None:
    checks = {c.name: c for c in call_checks([turn(), turn(bot_audio_s=0.0)], summary())}

    assert checks["call_bot_audio_every_turn"].status == "fail"
    assert checks["call_bot_audio_every_turn"].value == 1


def test_latency_above_the_target_warns() -> None:
    checks = {c.name: c for c in call_checks([turn()], summary(model_p95_s=1.8))}

    assert checks["call_model_p95"].status == "warn"
    assert checks["call_model_p50"].status == "pass"


def test_no_turn_released_by_smart_turn_fails_the_latency_checks() -> None:
    no_model = summary(model_turns=0, model_p50_s=None, model_p95_s=None)
    checks = {c.name: c for c in call_checks([turn()], no_model)}

    assert checks["call_model_p50"].status == "fail"
    assert checks["call_model_p95"].status == "fail"
    assert "no turn" in checks["call_model_p95"].detail


def test_no_turns_at_all_fails() -> None:
    checks = {c.name: c for c in call_checks([], summary())}

    assert checks["call_bot_audio_every_turn"].status == "fail"
