"""A call over Pipecat's SmallWebRTC transport (design D1, D6; ADR 0003).

The starter answers the client's offer with a new peer connection, builds the
call's pipeline over the shared voice models and runs it in the background. The
peer connection uses host candidates only (no STUN or TURN), as ADR 0003
measured from the browser to the container. Pipecat's own idle timeout is off:
the call registry ends idle calls, fed by the activity this observer reports.
The turn observer logs every turn's latency, sends it to the client as an RTVI
server message and logs the call summary when the pipeline ends.
"""

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pipecat.frames.frames import BotStartedSpeakingFrame, UserStartedSpeakingFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from parlora_core.session import SessionSettings
from parlora_server.adapters.stt.whisper import SAMPLE_RATE as STT_SAMPLE_RATE
from parlora_server.adapters.tts.kokoro import SAMPLE_RATE as TTS_SAMPLE_RATE
from parlora_server.config import Settings
from parlora_server.voice.metrics import LatencySink, TurnLatency, TurnMetricsObserver
from parlora_server.voice.pipeline import build_pipeline
from parlora_server.voice.prompts import load_prompt
from parlora_server.voice.registry import (
    CallHooks,
    IceCandidate,
    InvalidOfferError,
    SdpDescription,
)
from parlora_server.voice.runtime import VoiceRuntime

logger = logging.getLogger(__name__)

ACTIVITY_FRAMES = (UserStartedSpeakingFrame, BotStartedSpeakingFrame)


class ActivityObserver(BaseObserver):
    """Reports each time the learner or the tutor starts speaking."""

    def __init__(self, on_activity: Callable[[], None]) -> None:
        super().__init__()
        self._on_activity = on_activity

    async def on_push_frame(self, data: FramePushed) -> None:
        if isinstance(data.frame, ACTIVITY_FRAMES):
            self._on_activity()


class ServerMessages(Protocol):
    async def send_server_message(self, data: Any) -> None: ...


def latency_messages(rtvi: Callable[[], ServerMessages]) -> LatencySink:
    """Sends each turn's latency to the client through the worker's RTVI processor."""

    async def send(turn: TurnLatency) -> None:
        await rtvi().send_server_message(turn.client_message())

    return send


class WebRTCCall:
    def __init__(
        self, connection: SmallWebRTCConnection, worker: PipelineWorker, task: asyncio.Task[None]
    ) -> None:
        # Pipecat's connection methods are untyped.
        self._connection: Any = connection
        self._worker = worker
        self._task = task

    async def add_ice_candidates(self, candidates: Sequence[IceCandidate]) -> None:
        from aiortc.sdp import candidate_from_sdp

        for c in candidates:
            candidate: Any = None  # an empty line marks the end of candidates (RFC 8840)
            if c.candidate:
                candidate = candidate_from_sdp(c.candidate.removeprefix("candidate:"))
                candidate.sdpMid = c.sdp_mid
                candidate.sdpMLineIndex = c.sdp_mline_index
            await self._connection.add_ice_candidate(candidate)

    async def close(self) -> None:
        await self._worker.cancel(reason="call ended")
        await self._connection.disconnect()


class WebRTCCallStarter:
    """`CallStarter` over SmallWebRTC and the voice runtime's models."""

    def __init__(self, voice: VoiceRuntime, settings: Settings) -> None:
        self._voice = voice
        self._settings = settings

    async def __call__(
        self, offer: SdpDescription, settings: SessionSettings, hooks: CallHooks
    ) -> tuple[WebRTCCall, SdpDescription]:
        language = settings.target_language
        system_prompt = load_prompt(language)
        connection: Any = SmallWebRTCConnection()  # untyped methods
        try:
            await connection.initialize(sdp=offer.sdp, type=offer.type)
        except Exception as error:
            await connection.disconnect()
            # aiortc's message may quote the SDP: only its type is kept.
            raise InvalidOfferError(type(error).__name__) from None
        answer = connection.get_answer()
        # aiortc answers even an offer it cannot read, with no media in it.
        if answer is None or "m=audio" not in answer["sdp"]:
            await connection.disconnect()
            raise InvalidOfferError("the offer has no audio")

        transport = SmallWebRTCTransport(
            connection, TransportParams(audio_in_enabled=True, audio_out_enabled=True)
        )
        vad, turn_analyzer = self._voice.analysers()
        call = build_pipeline(
            transport_in=transport.input(),
            transport_out=transport.output(),
            models=self._voice.models(),
            settings=self._settings,
            language=language,
            system_prompt=system_prompt,
            vad=vad,
            turn_analyzer=turn_analyzer,
        )
        metrics = TurnMetricsObserver(on_latency=latency_messages(lambda: worker.rtvi))
        worker = PipelineWorker(
            call.pipeline,
            params=PipelineParams(
                audio_in_sample_rate=STT_SAMPLE_RATE,
                audio_out_sample_rate=TTS_SAMPLE_RATE,
                enable_metrics=True,
            ),
            idle_timeout_secs=None,
            observers=[ActivityObserver(hooks.activity), metrics],
        )

        async def on_client_disconnected(*_: Any) -> None:
            await worker.cancel(reason="client disconnected")

        transport.add_event_handler("on_client_disconnected", on_client_disconnected)

        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        task = asyncio.create_task(runner.run(), name="call pipeline")

        def on_done(_: asyncio.Task[None]) -> None:
            metrics.log_summary()
            hooks.ended()

        task.add_done_callback(on_done)
        return WebRTCCall(connection, worker, task), SdpDescription(answer["sdp"], answer["type"])
