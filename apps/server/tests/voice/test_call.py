"""The WebRTC call starter against a real aiortc offer, with fake models (no GPU).

The offer comes from an aiortc peer in the same process, with host candidates
only: nothing leaves the machine and no audio is sent.
"""

import asyncio
from typing import Any

import pytest
from aiortc import RTCPeerConnection
from fakes import FakeLanguageModel, FakeSpeechToText, FakeTextToSpeech
from pipecat.audio.turn.smart_turn.base_smart_turn import BaseSmartTurn, SmartTurnParams
from pipecat.frames.frames import BotStartedSpeakingFrame, TextFrame, UserStartedSpeakingFrame
from pipecat.observers.base_observer import FramePushed

from parlora_core.cefr import CefrLevel
from parlora_core.languages import LanguageCode
from parlora_core.learner import LearnerRef
from parlora_core.session import SessionSettings
from parlora_server.config import Settings
from parlora_server.voice.call import ActivityObserver, WebRTCCallStarter, latency_messages
from parlora_server.voice.metrics import TurnLatency
from parlora_server.voice.registry import (
    CallHooks,
    IceCandidate,
    InvalidOfferError,
    SdpDescription,
)
from parlora_server.voice.runtime import VoiceRuntime

pytestmark = pytest.mark.unit

EN = LanguageCode("en")
SESSION = SessionSettings(LearnerRef("ephemeral:test"), EN, EN, CefrLevel.B1, "Weekend plans")


class AlwaysComplete(BaseSmartTurn):
    def __init__(self) -> None:
        super().__init__(params=SmartTurnParams(stop_secs=3.0))

    def _predict_endpoint(self, audio_array: Any) -> dict[str, Any]:
        return {"prediction": 1, "probability": 1.0}


def fake_runtime() -> VoiceRuntime:
    components: dict[str, Any] = {
        "stt": FakeSpeechToText(),
        "llm": FakeLanguageModel(),
        "tts": FakeTextToSpeech(),
        "vad": lambda: (None, AlwaysComplete()),
    }

    def loader(name: str) -> Any:
        async def load() -> Any:
            return components[name]

        return load

    runtime = VoiceRuntime(enabled=True, loaders={name: loader(name) for name in components})
    asyncio.run(runtime.load())
    return runtime


async def browser_offer() -> tuple[RTCPeerConnection, SdpDescription]:
    peer = RTCPeerConnection()
    peer.addTransceiver("audio", direction="sendrecv")
    await peer.setLocalDescription(await peer.createOffer())
    local = peer.localDescription
    return peer, SdpDescription(local.sdp, local.type)


@pytest.mark.slow
def test_an_offer_is_answered_and_the_call_closes_cleanly() -> None:
    starter = WebRTCCallStarter(fake_runtime(), Settings(_env_file=None))
    ended = asyncio.Event()

    async def scenario() -> SdpDescription:
        peer, offer = await browser_offer()
        hooks = CallHooks(activity=lambda: None, ended=ended.set)
        try:
            call, answer = await starter(offer, SESSION, hooks)
            candidate = "candidate:1 1 udp 2122260223 127.0.0.1 50000 typ host"
            await call.add_ice_candidates(
                [IceCandidate(candidate, "0", 0), IceCandidate("", "0", 0)]
            )
            await call.close()
            await asyncio.wait_for(ended.wait(), timeout=10)
        finally:
            await peer.close()
        return answer

    answer = asyncio.run(scenario())

    assert answer.type == "answer"
    assert "m=audio" in answer.sdp


def test_an_unusable_offer_is_invalid_without_quoting_it() -> None:
    starter = WebRTCCallStarter(fake_runtime(), Settings(_env_file=None))
    hooks = CallHooks(activity=lambda: None, ended=lambda: None)

    with pytest.raises(InvalidOfferError) as error:
        asyncio.run(starter(SdpDescription("SECRET-SDP-MARKER", "offer"), SESSION, hooks))

    assert "SECRET-SDP-MARKER" not in str(error.value)


def test_the_activity_observer_reports_speech_starts_only() -> None:
    seen: list[str] = []
    observer = ActivityObserver(lambda: seen.append("activity"))

    async def push(frame: Any) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=None,  # type: ignore[arg-type]
                destination=None,  # type: ignore[arg-type]
                frame=frame,
                direction=None,  # type: ignore[arg-type]
                timestamp=0,
            )
        )

    async def scenario() -> None:
        await push(UserStartedSpeakingFrame())
        await push(TextFrame("hello"))
        await push(BotStartedSpeakingFrame())

    asyncio.run(scenario())
    assert seen == ["activity", "activity"]


def test_turn_latency_is_sent_to_the_client_as_a_server_message() -> None:
    class Rtvi:
        def __init__(self) -> None:
            self.sent: list[Any] = []

        async def send_server_message(self, data: Any) -> None:
            self.sent.append(data)

    rtvi = Rtvi()
    send = latency_messages(lambda: rtvi)

    asyncio.run(send(TurnLatency(0.9, 0.1, 0.3, 0.2, interrupted=False)))

    assert rtvi.sent == [
        {
            "type": "turn-latency",
            "voice_to_voice_ms": 900,
            "stt_ms": 100,
            "llm_ttft_ms": 300,
            "tts_first_audio_ms": 200,
            "interrupted": False,
        }
    ]
