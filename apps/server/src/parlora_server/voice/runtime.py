"""The voice models of the server, loaded and warmed up at start-up (design D7).

Each voice dependency (`stt`, `tts`, `llm`, `vad`) has a loader that loads it and
runs one warm-up inference, so the first turn of the first call does not pay for
a cold start (ADR 0009: 5-15 s). Loaders run one after another in the background
while the server already answers liveness; readiness reports each one as
`loading`, `ok` or `failed`, or `disabled` when voice is off (no GPU, CI).
Failures are logged with the dependency's name and the adapter's message, which
names models and endpoints, never a conversation.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer
from pipecat.audio.vad.vad_analyzer import VADAnalyzer

from parlora_core.audio import AudioChunk
from parlora_core.languages import LanguageCode
from parlora_core.ports import LanguageModel, SpeechToText, TextToSpeech
from parlora_server.config import Settings
from parlora_server.voice.pipeline import VoiceModels, silero_vad, smart_turn

logger = logging.getLogger(__name__)

CheckState = Literal["ok", "loading", "failed", "disabled"]
VOICE_CHECKS = ("stt", "tts", "llm", "vad")
Loader = Callable[[], Awaitable[Any]]
# Per call: a fresh VAD and turn analyser, since both keep the state of the audio.
Analysers = Callable[[], tuple[VADAnalyzer, BaseTurnAnalyzer]]

WARM_UP_LANGUAGE = LanguageCode("en")


class VoiceRuntime:
    def __init__(self, *, enabled: bool, loaders: Mapping[str, Loader]) -> None:
        missing = [name for name in VOICE_CHECKS if name not in loaders]
        if enabled and missing:
            raise ValueError(f"no loader for {', '.join(missing)}")
        self.enabled = enabled
        self._loaders = dict(loaders)
        self._states: dict[str, CheckState] = dict.fromkeys(
            VOICE_CHECKS, "loading" if enabled else "disabled"
        )
        self._components: dict[str, Any] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "VoiceRuntime":
        return cls(enabled=settings.voice.enabled, loaders=default_loaders(settings))

    def state(self, name: str) -> CheckState:
        return self._states[name]

    def check(self, name: str) -> Callable[[], Awaitable[str]]:
        """A readiness check reporting this dependency's state."""

        async def check() -> str:
            return self._states[name]

        return check

    def unavailable(self) -> list[str]:
        """The voice dependencies a call cannot use yet, in check order."""
        return [name for name in VOICE_CHECKS if self._states[name] != "ok"]

    def component(self, name: str) -> Any:
        try:
            return self._components[name]
        except KeyError:
            raise LookupError(f"{name} is not loaded") from None

    def models(self) -> VoiceModels:
        return VoiceModels(
            stt=self.component("stt"), llm=self.component("llm"), tts=self.component("tts")
        )

    def analysers(self) -> tuple[VADAnalyzer, BaseTurnAnalyzer]:
        factory: Analysers = self.component("vad")
        return factory()

    async def load(self) -> None:
        if not self.enabled:
            return
        for name in VOICE_CHECKS:
            try:
                self._components[name] = await self._loaders[name]()
            except Exception as error:
                self._states[name] = "failed"
                logger.error("voice: %s failed to load: %s", name, error)
            else:
                self._states[name] = "ok"
                logger.info("voice: %s loaded and warmed up", name)

    async def aclose(self) -> None:
        for component in self._components.values():
            close = getattr(component, "aclose", None)
            if close is not None:
                await close()


# ── The real loaders ────────────────────────────────────────────────────────


def default_loaders(settings: Settings) -> dict[str, Loader]:
    async def stt() -> SpeechToText:
        from parlora_server.adapters.stt.whisper import SAMPLE_RATE, load_whisper

        model = await asyncio.to_thread(load_whisper, settings.stt)
        await model.transcribe(
            AudioChunk(b"\0\0" * SAMPLE_RATE, SAMPLE_RATE), language=WARM_UP_LANGUAGE
        )
        return model

    async def tts() -> TextToSpeech:
        from parlora_server.adapters.tts.kokoro import Kokoro, KokoroTextToSpeech

        directory = Path(settings.models_dir) / "kokoro"
        kokoro = await asyncio.to_thread(
            Kokoro, directory, settings.tts.voice, provider=settings.tts.provider
        )
        model = KokoroTextToSpeech(kokoro)
        async for _ in model.stream_speech(
            "Hello there.", language=WARM_UP_LANGUAGE, voice=settings.tts.voice
        ):
            pass
        return model

    async def llm() -> LanguageModel:
        from parlora_server.adapters.llm.ollama import load_ollama

        model = load_ollama(settings.llm)
        try:
            await model.warm_up()
        except BaseException:
            await model.aclose()
            raise
        return model

    async def vad() -> Analysers:
        def analysers() -> tuple[VADAnalyzer, BaseTurnAnalyzer]:
            return silero_vad(settings.vad), smart_turn(settings.vad)

        # Building them once downloads and loads both ONNX models.
        await asyncio.to_thread(analysers)
        return analysers

    return {"stt": stt, "tts": tts, "llm": llm, "vad": vad}
