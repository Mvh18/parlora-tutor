"""The call pipeline with fake adapters and a scripted turn analyser (no models, no GPU)."""

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from fakes import FakeLanguageModel, FakeSpeechToText, FakeTextToSpeech
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    TTSAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from parlora_core.audio import AudioChunk
from parlora_core.languages import LanguageCode
from parlora_core.ports import Message
from parlora_server.config import Settings, VadSettings
from parlora_server.voice import pipeline as voice

pytestmark = pytest.mark.unit

EN = LanguageCode("en")
RATE = 16_000
FRAME_MS = 20
PROMPT = "You are a friendly English conversation partner."


class Passthrough(FrameProcessor):
    """Stands in for the WebRTC transport: frames go straight through."""

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class ScriptedTurn(BaseSmartTurn):
    """Smart Turn with scripted verdicts instead of the model (1 complete, 0 incomplete)."""

    def __init__(self, verdicts: Sequence[int], fallback_secs: float = 3.0) -> None:
        super().__init__(params=SmartTurnParams(stop_secs=fallback_secs))
        self.verdicts = list(verdicts)
        self.asked = 0

    def _predict_endpoint(self, audio_array: Any) -> dict[str, Any]:
        self.asked += 1
        verdict = self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]
        return {"prediction": verdict, "probability": float(verdict)}


def audio(ms: int, loud: bool) -> list[Frame]:
    samples = RATE * FRAME_MS // 1000
    level = 3_000 if loud else 0
    chunk = (np.full(samples, level, dtype=np.int16)).tobytes()
    return [InputAudioRawFrame(chunk, RATE, 1) for _ in range(ms // FRAME_MS)]


def utterance(speech_ms: int = 600) -> list[Frame]:
    return [
        VADUserStartedSpeakingFrame(start_secs=0.2),
        *audio(speech_ms, loud=True),
        VADUserStoppedSpeakingFrame(stop_secs=0.2),
    ]


@dataclass
class Call:
    stt: FakeSpeechToText = field(default_factory=FakeSpeechToText)
    llm: FakeLanguageModel = field(default_factory=FakeLanguageModel)
    tts: FakeTextToSpeech = field(default_factory=FakeTextToSpeech)
    turn: ScriptedTurn = field(default_factory=lambda: ScriptedTurn([1]))

    def run(self, frames: Sequence[Frame]) -> Sequence[Frame]:
        built = voice.build_pipeline(
            transport_in=Passthrough(),
            transport_out=Passthrough(),
            models=voice.VoiceModels(self.stt, self.llm, self.tts),
            settings=Settings(_env_file=None),
            language=EN,
            system_prompt=PROMPT,
            vad=None,
            turn_analyzer=self.turn,
        )
        params = PipelineParams(audio_in_sample_rate=RATE, audio_out_sample_rate=24_000)
        down, _ = asyncio.run(
            run_test(built.pipeline, frames_to_send=frames, pipeline_params=params)
        )
        return down


def test_one_turn_goes_from_speech_to_bot_audio() -> None:
    call = Call()

    down = call.run([*utterance(), *audio(200, loud=False), SleepFrame(0.5)])

    assert len(call.stt.calls) == 1, "the utterance is transcribed once"
    assert call.stt.calls[0].sample_rate == RATE
    assert call.llm.calls == [
        [Message("system", PROMPT), Message("user", "I went to the market yesterday.")]
    ]
    assert call.tts.texts, "the reply reaches text-to-speech"
    assert any(isinstance(f, TTSAudioRawFrame) for f in down), "bot audio reaches the output"


def test_a_pause_inside_a_sentence_stays_one_turn() -> None:
    call = Call(turn=ScriptedTurn([0, 1]))
    texts = iter(["Yesterday I went", "to the market."])

    async def transcribe(audio: AudioChunk, *, language: LanguageCode) -> Any:
        call.stt.calls.append(audio)
        from parlora_core.audio import Transcript

        return Transcript(next(texts), language, audio.duration_s, 0.01)

    call.stt.transcribe = transcribe  # type: ignore[method-assign]

    call.run(
        [
            *utterance(),
            *audio(400, loud=False),  # the learner pauses to think
            SleepFrame(0.2),
            *utterance(),
            *audio(200, loud=False),
            SleepFrame(0.5),
        ]
    )

    assert len(call.llm.calls) == 1, "one user turn"
    (user,) = [m for m in call.llm.calls[0] if m.role == "user"]
    assert "Yesterday I went" in user.content
    assert "to the market." in user.content


def test_the_turn_ends_after_the_fallback_timeout_when_the_analyser_is_unsure() -> None:
    call = Call(turn=ScriptedTurn([0], fallback_secs=0.3))

    call.run([*utterance(), *audio(400, loud=False), SleepFrame(0.5)])

    assert len(call.llm.calls) == 1, "silence beyond the fallback ends the turn"


@dataclass
class SlowTextToSpeech(FakeTextToSpeech):
    """Speaks each clause as ten chunks, 50 ms apart, so a learner can talk over it."""

    chunks_sent: int = 0

    async def stream_speech(
        self, text: str, *, language: LanguageCode, voice: str
    ) -> AsyncIterator[AudioChunk]:
        self.texts.append(text)
        for _ in range(10):
            await asyncio.sleep(0.05)
            self.chunks_sent += 1
            yield self.audio


def test_learner_speech_interrupts_the_bot_and_stops_its_audio() -> None:
    tts = SlowTextToSpeech()
    llm = FakeLanguageModel(
        chunks=["Well,", " that", " is", " a", " long", " story.", " Tell", " me", " more."]
    )
    call = Call(llm=llm, tts=tts)

    down = call.run(
        [
            *utterance(),
            *audio(200, loud=False),
            SleepFrame(0.3),  # the bot starts speaking
            VADUserStartedSpeakingFrame(start_secs=0.2),  # the learner talks over it
            SleepFrame(0.8),
        ]
    )

    # The learner's first turn also starts with an interruption; the barge-in is the last.
    interruptions = [i for i, f in enumerate(down) if isinstance(f, InterruptionFrame)]
    assert len(interruptions) >= 2, "an interruption is raised while the bot speaks"
    barge_in = interruptions[-1]
    assert any(isinstance(f, TTSAudioRawFrame) for f in down[:barge_in]), "the bot was speaking"
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down[barge_in + 1 :]), (
        "no bot audio after the interruption"
    )
    assert tts.chunks_sent < 20, "the rest of the reply is not synthesised"


def test_vad_and_turn_settings_reach_the_analysers() -> None:
    settings = VadSettings(confidence=0.5, start_secs=0.1, stop_secs=0.3, min_volume=0.4)

    params = voice.vad_params(settings)

    assert (params.confidence, params.start_secs, params.stop_secs, params.min_volume) == (
        0.5,
        0.1,
        0.3,
        0.4,
    )
