"""The turn observer over recorded frames, with a fake clock (no pipeline, no models)."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import pytest
from fakes import FakeLanguageModel, FakeSpeechToText, FakeTextToSpeech
from hypothesis import given
from hypothesis import strategies as st
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from parlora_core.languages import LanguageCode
from parlora_server.voice.metrics import (
    LATENCY_MESSAGE_TYPE,
    TurnLatency,
    TurnMetricsObserver,
    percentile,
    summarise,
)
from parlora_server.voice.services import (
    LanguageModelService,
    SpeechToTextService,
    TextToSpeechService,
)

pytestmark = pytest.mark.unit

EN = LanguageCode("en")
LEARNER_TEXT = "I went to the market yesterday."
TUTOR_TEXT = "Nice! What did you buy?"
LOGGER = "parlora_server.voice.metrics"


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


class Sources:
    def __init__(self) -> None:
        self.other = FrameProcessor()
        self.stt = SpeechToTextService(FakeSpeechToText(), language=EN)
        self.llm = LanguageModelService(FakeLanguageModel())
        self.tts = TextToSpeechService(FakeTextToSpeech(), language=EN, voice="af_heart")


SOURCES = Sources()


def push(
    observer: TurnMetricsObserver,
    frame: Frame,
    source: FrameProcessor | None = None,
    direction: FrameDirection = FrameDirection.DOWNSTREAM,
) -> None:
    data = FramePushed(
        source=source or SOURCES.other,
        destination=SOURCES.other,
        frame=frame,
        direction=direction,
        timestamp=0,
    )
    asyncio.run(observer.on_push_frame(data))


def one_turn(
    observer: TurnMetricsObserver,
    clock: Clock,
    *,
    latency: float = 0.9,
    stop_secs: float = 0.2,
) -> None:
    """The frames of one completed turn; the learner stops speaking at `clock.now`."""
    stopped_at = clock.now
    push(observer, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=stopped_at - 1))
    push(
        observer, VADUserStoppedSpeakingFrame(stop_secs=stop_secs, timestamp=stopped_at + stop_secs)
    )
    push(observer, UserStoppedSpeakingFrame())
    push(observer, TranscriptionFrame(LEARNER_TEXT, "learner", "now"), SOURCES.stt)
    stt = ProcessingMetricsData(processor="stt", value=0.12)
    push(observer, MetricsFrame(data=[stt]), SOURCES.stt)
    push(observer, MetricsFrame(data=[TTFBMetricsData(processor="llm", value=0.3)]), SOURCES.llm)
    push(observer, LLMTextFrame(TUTOR_TEXT), SOURCES.llm)
    push(observer, MetricsFrame(data=[TTFBMetricsData(processor="tts", value=0.25)]), SOURCES.tts)
    clock.now = stopped_at + latency
    push(observer, BotStartedSpeakingFrame(), direction=FrameDirection.UPSTREAM)
    push(observer, BotStartedSpeakingFrame())
    clock.now += 2
    push(observer, BotStoppedSpeakingFrame())
    clock.now += 1


def turn_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and hasattr(r, "turn_latency")]


def test_one_completed_turn_gives_one_info_record_with_the_latencies_and_no_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    one_turn(observer, clock, latency=0.9)

    (record,) = turn_records(caplog)
    assert record.levelno == logging.INFO
    assert record.turn_latency == {  # type: ignore[attr-defined]
        "voice_to_voice_ms": 900,
        "stt_ms": 120,
        "llm_ttft_ms": 300,
        "tts_first_audio_ms": 250,
        "interrupted": False,
    }
    assert "voice_to_voice_ms=900" in record.getMessage()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "market" not in logged, "no learner text at INFO"
    assert "buy" not in logged, "no tutor text at INFO"


def test_the_turn_text_is_logged_only_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    one_turn(observer, clock)

    with_text = [r for r in caplog.records if "market" in r.getMessage()]
    assert with_text
    assert all(r.levelno == logging.DEBUG for r in with_text)
    assert any(TUTOR_TEXT in r.getMessage() for r in with_text)


def test_voice_to_voice_starts_when_the_learner_fell_silent_not_when_the_vad_decided() -> None:
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    one_turn(observer, clock, latency=1.1, stop_secs=0.4)

    assert observer.turns[0].voice_to_voice_s == pytest.approx(1.1)


def test_a_pause_inside_the_turn_measures_from_the_last_stop() -> None:
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    push(observer, VADUserStoppedSpeakingFrame(stop_secs=0.0, timestamp=clock.now - 3))
    push(observer, VADUserStartedSpeakingFrame(timestamp=clock.now - 2))  # still the same turn
    one_turn(observer, clock, latency=0.8)

    assert len(observer.turns) == 1
    assert observer.turns[0].voice_to_voice_s == pytest.approx(0.8)
    assert not observer.turns[0].interrupted


def test_barge_in_logs_the_turn_as_interrupted(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    push(observer, VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=clock.now + 0.2))
    push(observer, UserStoppedSpeakingFrame())
    clock.now += 1.0
    push(observer, BotStartedSpeakingFrame())
    push(observer, VADUserStartedSpeakingFrame())  # the learner talks over the tutor
    push(observer, BotStoppedSpeakingFrame())  # the interrupted reply's audio stops

    (turn,) = observer.turns
    assert turn.interrupted
    assert turn.voice_to_voice_s == pytest.approx(1.0)
    (record,) = turn_records(caplog)
    assert record.turn_latency["interrupted"] is True  # type: ignore[attr-defined]


def test_a_reply_interrupted_before_its_audio_has_no_voice_to_voice_latency() -> None:
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    push(observer, VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=clock.now))
    push(observer, UserStoppedSpeakingFrame())
    push(observer, VADUserStartedSpeakingFrame())

    (turn,) = observer.turns
    assert turn.interrupted
    assert turn.voice_to_voice_s is None


def test_frames_outside_a_turn_and_zero_ttfb_are_ignored() -> None:
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)

    push(observer, BotStartedSpeakingFrame())  # a greeting: no learner turn yet
    push(observer, BotStoppedSpeakingFrame())
    push(observer, VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=clock.now))
    push(observer, UserStoppedSpeakingFrame())
    zero = TTFBMetricsData(processor="llm", value=0.0)  # a cancelled TTFB measurement
    push(observer, MetricsFrame(data=[zero]), SOURCES.llm)
    push(observer, MetricsFrame(data=[TTFBMetricsData(processor="x", value=5.0)]))
    push(observer, BotStoppedSpeakingFrame())  # not speaking yet: the turn goes on

    assert observer.turns == []
    clock.now += 1
    push(observer, BotStartedSpeakingFrame())
    push(observer, BotStoppedSpeakingFrame())
    (turn,) = observer.turns
    assert turn.llm_ttft_s is None
    assert turn.stt_s is None


def test_the_last_turn_latency_goes_to_the_client_when_the_bot_starts_speaking() -> None:
    clock = Clock()
    sent: list[dict[str, Any]] = []

    async def on_latency(turn: TurnLatency) -> None:
        sent.append(turn.client_message())

    observer = TurnMetricsObserver(on_latency=on_latency, clock=clock)

    one_turn(observer, clock, latency=0.9)

    assert sent == [
        {
            "type": LATENCY_MESSAGE_TYPE,
            "voice_to_voice_ms": 900,
            "stt_ms": 120,
            "llm_ttft_ms": 300,
            "tts_first_audio_ms": 250,
            "interrupted": False,
        }
    ]


def test_a_call_with_ten_turns_logs_the_count_p50_and_p95(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    clock = Clock()
    observer = TurnMetricsObserver(clock=clock)
    for latency in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
        one_turn(observer, clock, latency=latency)

    summary = observer.log_summary()

    assert (summary.turns, summary.measured, summary.interrupted) == (10, 10, 0)
    assert summary.p50_s == pytest.approx(0.95)
    assert summary.p95_s == pytest.approx(1.355)
    (record,) = [r for r in caplog.records if hasattr(r, "call_summary")]
    assert record.levelno == logging.INFO
    assert "turns=10" in record.getMessage()
    assert "p50_ms=950" in record.getMessage()
    assert "p95_ms=1355" in record.getMessage()


def test_a_call_without_measured_turns_has_no_percentiles(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    summary = TurnMetricsObserver().log_summary()

    assert (summary.turns, summary.p50_s, summary.p95_s) == (0, None, None)
    assert "p50_ms=-" in caplog.records[-1].getMessage()


def test_interrupted_turns_without_audio_are_counted_but_not_measured() -> None:
    turns = [
        TurnLatency(1.0, None, None, None, interrupted=False),
        TurnLatency(None, None, None, None, interrupted=True),
    ]

    summary = summarise(turns)

    assert (summary.turns, summary.measured, summary.interrupted) == (2, 1, 1)
    assert summary.p50_s == summary.p95_s == 1.0


def test_percentile_needs_values() -> None:
    with pytest.raises(ValueError, match="no values"):
        percentile([], 50)


@given(st.lists(st.floats(0, 10), min_size=1, max_size=50), st.floats(0, 100))
def test_percentile_lies_between_the_extremes_and_grows_with_q(
    values: Sequence[float], q: float
) -> None:
    value = percentile(values, q)

    assert min(values) <= value <= max(values)
    assert percentile(values, 0) == min(values)
    assert percentile(values, 100) == max(values)
    assert percentile(values, q / 2) <= value + 1e-9
