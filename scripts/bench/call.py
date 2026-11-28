"""Smoke-call benchmark: the synthetic fixtures played as learner turns of a real call.

The server's voice runtime loads the real models (faster-whisper, Kokoro, Silero
VAD with Smart Turn v3, and the host's Ollama) and its call starter answers an
offer from an aiortc peer in the same process, the stand-in for the browser:
host candidates only, nothing leaves the machine, so no ICE range is needed.
Each fixture in `fixtures/synthetic/stt/` is played into the call as one learner
turn; the client waits until the tutor's audio has been quiet for `--gap`
seconds (or `--turn-timeout` passes) before the next one. A data channel sends
the RTVI `client-ready` message, as the web client does.

The result records, per turn, the fixture id, how much bot audio arrived and
when the first audio arrived after the utterance ended (client side), the
server's turn records and its call summary, both taken from the turn observer's
log records. No transcript, reply text or audio is kept. Checks: every turn got
bot audio (fails otherwise), and the Phase 1 target over the turns Smart Turn
released (p50 ≤ 1.2 s, p95 ≤ 1.5 s; warns when exceeded, fails with no such turn).

Needs the GPU and Ollama on the host; the stack (`just up`) should be stopped so
the server's models are not loaded twice.

Usage: python -m scripts.bench.call [--gap S] [--turn-timeout S] [--output-dir DIR]
"""

import argparse
import asyncio
import fractions
import json
import logging
import os
import platform
import sys
import time
import uuid
import wave
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from scripts.bench.common.budgets import exit_code
from scripts.bench.common.call import (
    DEFAULT_THRESHOLD,
    BotAudioTracker,
    ClientTurn,
    call_checks,
    call_measurements,
)
from scripts.bench.common.results import RESULTS_DIR, write_result
from scripts.bench.common.schema import build_result
from scripts.bench.host import OLLAMA, collect_machine, ollama_version

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic" / "stt"
RTVI_LABEL = "rtvi-ai"
SEND_RATE = 48_000
FRAME_SAMPLES = 960  # 20 ms at 48 kHz, as a browser sends
DEFAULT_GAP_S = 2.0
DEFAULT_TURN_TIMEOUT_S = 30.0
SETTLE_S = 2.0  # lets the pipeline start before the first turn
PACKAGES = ("pipecat-ai", "aiortc", "faster-whisper", "onnxruntime-gpu", "onnxruntime")


class _Captured(logging.Handler):
    """Keeps the turn observer's `turn_latency` and `call_summary` record attributes."""

    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.turns: list[dict[str, Any]] = []
        self.summary: dict[str, Any] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        if hasattr(record, "turn_latency"):
            self.turns.append(dict(record.turn_latency))
        if hasattr(record, "call_summary"):
            self.summary = dict(record.call_summary)


def load_fixture(path: Path) -> tuple[bytes, float]:
    """The fixture as 48 kHz mono 16-bit PCM, and its duration in seconds."""
    import numpy as np

    with wave.open(str(path), "rb") as file:
        if file.getnchannels() != 1 or file.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected mono 16-bit audio")
        rate = file.getframerate()
        samples = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16)
    duration = len(samples) / rate
    times = np.arange(round(duration * SEND_RATE)) / SEND_RATE
    resampled = np.interp(times, np.arange(len(samples)) / rate, samples.astype(np.float64))
    return resampled.round().astype(np.int16).tobytes(), duration


def fixture_ids() -> list[str]:
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    return [u["id"] for u in manifest["utterances"]]


def _player() -> Any:
    """An audio track that plays queued PCM in real time and silence otherwise."""
    import av
    from aiortc import MediaStreamTrack

    class FixturePlayer(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self._pending = bytearray()
            self._played = asyncio.Event()
            self._played.set()
            self._start: float | None = None
            self._timestamp = 0

        def play(self, pcm: bytes) -> None:
            self._pending += pcm
            self._played.clear()

        async def wait_played(self) -> None:
            await self._played.wait()

        async def recv(self) -> Any:
            if self._start is None:
                self._start = time.monotonic()
            else:
                self._timestamp += FRAME_SAMPLES
                delay = self._start + self._timestamp / SEND_RATE - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
            size = FRAME_SAMPLES * 2
            chunk = bytes(self._pending[:size]).ljust(size, b"\0")
            del self._pending[:size]
            if not self._pending:
                self._played.set()
            frame = av.AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
            frame.planes[0].update(chunk)
            frame.sample_rate = SEND_RATE
            frame.pts = self._timestamp
            frame.time_base = fractions.Fraction(1, SEND_RATE)
            return frame

    return FixturePlayer()


async def _receive(track: Any, tracker: BotAudioTracker) -> None:
    from aiortc.mediastreams import MediaStreamError

    while True:
        try:
            frame = await track.recv()
        except MediaStreamError:
            return
        tracker.frame(
            frame.to_ndarray().tobytes(),
            duration_s=frame.samples / frame.sample_rate,
            at=time.monotonic(),
        )


async def _play_turn(
    fixture: str, player: Any, tracker: BotAudioTracker, gap_s: float, timeout_s: float
) -> ClientTurn:
    pcm, duration = load_fixture(FIXTURES / f"{fixture}.wav")
    tracker.start_turn(at=time.monotonic())
    player.play(pcm)
    await player.wait_played()
    ended = time.monotonic()
    while time.monotonic() < ended + timeout_s:
        quiet = tracker.quiet_for(time.monotonic())
        if quiet is not None and quiet >= gap_s:
            break
        await asyncio.sleep(0.05)
    first = None if tracker.first_loud_at is None else max(0.0, tracker.first_loud_at - ended)
    return ClientTurn(fixture, round(duration, 3), round(tracker.loud_s, 3), first)


def _bench_models_dir() -> Path:
    from scripts.bench.kokoro_runner import cache_dir

    return cache_dir(os.environ).parent


async def run_call(
    fixtures: Sequence[str], *, gap_s: float, timeout_s: float, threshold: float
) -> tuple[list[ClientTurn], _Captured, Any]:
    from aiortc import RTCPeerConnection, RTCSessionDescription

    from parlora_core.cefr import CefrLevel
    from parlora_core.languages import LanguageCode
    from parlora_core.learner import LearnerRef
    from parlora_core.session import SessionSettings
    from parlora_server.config import Settings
    from parlora_server.voice.call import WebRTCCallStarter
    from parlora_server.voice.registry import CallHooks, SdpDescription
    from parlora_server.voice.runtime import VoiceRuntime

    settings = Settings(
        _env_file=None,
        models_dir=_bench_models_dir(),
        llm={"base_url": OLLAMA},
        voice={"enabled": True, "ice_host": None},
    )
    captured = _Captured()
    metrics_logger = logging.getLogger("parlora_server.voice.metrics")
    metrics_logger.addHandler(captured)
    metrics_logger.setLevel(logging.INFO)
    runtime = VoiceRuntime.from_settings(settings)
    peer = RTCPeerConnection()
    receivers: list[asyncio.Task[None]] = []
    try:
        await runtime.load()
        if missing := runtime.unavailable():
            raise RuntimeError(f"voice dependencies not loaded: {', '.join(missing)}")
        tracker = BotAudioTracker(threshold)
        player = _player()
        peer.addTrack(player)
        channel = peer.createDataChannel("chat", ordered=True)

        @channel.on("open")
        def client_ready() -> None:
            about = {"version": "1.0.0", "about": {"library": "parlora-bench"}}
            message = {"label": RTVI_LABEL, "type": "client-ready", "id": str(uuid.uuid4())}
            channel.send(json.dumps({**message, "data": about}))

        @peer.on("track")
        def on_track(track: Any) -> None:
            receivers.append(asyncio.create_task(_receive(track, tracker)))

        await peer.setLocalDescription(await peer.createOffer())
        offer = SdpDescription(peer.localDescription.sdp, peer.localDescription.type)
        session = SessionSettings(
            LearnerRef("ephemeral:bench"),
            LanguageCode("en"),
            LanguageCode("en"),
            CefrLevel.B1,
            "Everyday life",
        )
        ended = asyncio.Event()
        call, answer = await WebRTCCallStarter(runtime, settings)(
            offer, session, CallHooks(activity=lambda: None, ended=ended.set)
        )
        await peer.setRemoteDescription(RTCSessionDescription(answer.sdp, answer.type))
        await asyncio.sleep(SETTLE_S)
        turns = [
            await _play_turn(fixture, player, tracker, gap_s, timeout_s) for fixture in fixtures
        ]
        await call.close()
        await asyncio.wait_for(ended.wait(), timeout=30)
    finally:
        await peer.close()
        for task in receivers:
            task.cancel()
        await runtime.aclose()
        metrics_logger.removeHandler(captured)
    return turns, captured, settings


def versions() -> dict[str, Any]:
    found: dict[str, Any] = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    found["ollama"] = ollama_version()
    return found


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP_S, help="quiet seconds")
    parser.add_argument("--turn-timeout", type=float, default=DEFAULT_TURN_TIMEOUT_S)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--variant", default="native")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    fixtures = fixture_ids()
    turns, captured, settings = asyncio.run(
        run_call(fixtures, gap_s=args.gap, timeout_s=args.turn_timeout, threshold=args.threshold)
    )
    if captured.summary is None:
        print("call: the server logged no call summary", file=sys.stderr)
        return 1
    checks = call_checks(turns, captured.summary)
    result = build_result(
        component="call",
        variant=args.variant,
        machine=collect_machine(),
        software=versions(),
        settings={
            "llm_model": settings.llm.model,
            "stt_model": settings.stt.model,
            "stt_beam_size": settings.stt.beam_size,
            "tts_provider": settings.tts.provider,
            "tts_voice": settings.tts.voice,
            "vad_stop_secs": settings.vad.stop_secs,
            "turn_fallback_secs": settings.vad.turn_fallback_secs,
            "fixtures": len(fixtures),
            "quiet_gap_s": args.gap,
            "turn_timeout_s": args.turn_timeout,
            "loudness_threshold": args.threshold,
        },
        measurements=call_measurements(turns, captured.turns, captured.summary),
        checks=[check.to_dict() for check in checks],
    )
    path = write_result(result, args.output_dir)
    for check in checks:
        print(f"call: {check.name} {check.status} ({check.value} / {check.limit} {check.unit})")
    print(f"call: wrote {path}")
    return exit_code(checks)


if __name__ == "__main__":
    sys.exit(main())
