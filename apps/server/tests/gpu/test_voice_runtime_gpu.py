"""Start-up loading and warm-up of every voice dependency for real. Needs the GPU: never in CI."""

import asyncio
import importlib
import os
import sys
from pathlib import Path

import httpx2
import pytest

from parlora_server.config import Settings
from parlora_server.voice.runtime import VOICE_CHECKS, VoiceRuntime

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

LOCAL_OLLAMA = "http://127.0.0.1:11434"


def bench_models_dir() -> Path:
    """The benchmarks' weight cache, so the test downloads nothing new (Kokoro in `kokoro/`)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "bench"))
    runner = importlib.import_module("kokoro_runner")
    return Path(runner.cache_dir(os.environ)).parent


def test_every_voice_dependency_loads_and_warms_up() -> None:
    try:
        httpx2.get(LOCAL_OLLAMA + "/api/version", timeout=2).raise_for_status()
    except httpx2.HTTPError:
        pytest.skip("Ollama is not running on the host")
    settings = Settings(
        _env_file=None,
        models_dir=bench_models_dir(),
        llm={"base_url": LOCAL_OLLAMA},
    )
    runtime = VoiceRuntime.from_settings(settings)

    async def scenario() -> None:
        try:
            await runtime.load()
            assert runtime.unavailable() == []
            vad, turn = runtime.analysers()
            assert vad is not None
            assert turn is not None
            runtime.models()
        finally:
            await runtime.aclose()

    asyncio.run(scenario())
    assert {runtime.state(name) for name in VOICE_CHECKS} == {"ok"}
