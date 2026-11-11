"""Voice dependencies in readiness, loaded and warmed up at start-up (design D7)."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from parlora_core.errors import LanguageModelError
from parlora_server.config import Settings
from parlora_server.main import create_app
from parlora_server.voice.runtime import VOICE_CHECKS, VoiceRuntime

pytestmark = pytest.mark.unit


async def database_ok() -> str:
    return "ok"


async def loaded() -> object:
    return object()


async def never_loads() -> object:
    await asyncio.Event().wait()
    raise AssertionError("unreachable")


async def ollama_missing() -> object:
    raise LanguageModelError("qwen3: Ollama answered HTTP 404: model not found")


def runtime(**overrides: Callable[[], Any]) -> VoiceRuntime:
    loaders: dict[str, Callable[[], Any]] = dict.fromkeys(VOICE_CHECKS, loaded)
    loaders.update(overrides)
    return VoiceRuntime(enabled=True, loaders=loaders)


def ready(client: TestClient) -> Any:
    """Readiness once no check is loading any more (or still loading after a while)."""
    response = client.get("/api/v1/health/ready")
    for _ in range(200):
        if "loading" not in response.json()["checks"].values():
            break
        response = client.get("/api/v1/health/ready")
    return response


def test_voice_disabled_keeps_database_only_readiness(
    make_settings: Callable[..., Settings],
) -> None:
    app = create_app(make_settings(voice={"enabled": False}))
    app.state.readiness_checks["database"] = database_ok

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {
        "database": "ok",
        "stt": "disabled",
        "tts": "disabled",
        "llm": "disabled",
        "vad": "disabled",
    }


def test_ready_once_every_voice_dependency_is_warmed_up(settings: Settings) -> None:
    app = create_app(settings, voice=runtime())
    app.state.readiness_checks["database"] = database_ok

    with TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 200
    assert response.json()["checks"] == dict.fromkeys(["database", *VOICE_CHECKS], "ok")


def test_not_ready_while_speech_to_text_is_loading(settings: Settings) -> None:
    app = create_app(settings, voice=runtime(stt=never_loads))
    app.state.readiness_checks["database"] = database_ok

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["checks"]["stt"] == "loading"


def test_a_dependency_that_fails_to_load_is_failed_and_logged_without_content(
    settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    app = create_app(settings, voice=runtime(llm=ollama_missing))
    app.state.readiness_checks["database"] = database_ok

    with caplog.at_level(logging.ERROR), TestClient(app) as client:
        response = ready(client)

    assert response.status_code == 503
    checks = response.json()["checks"]
    assert checks["llm"] == "failed"
    assert checks["stt"] == checks["tts"] == checks["vad"] == "ok"
    assert any(
        "llm" in r.getMessage() and "model not found" in r.getMessage() for r in caplog.records
    )


def test_runtime_reports_what_is_unavailable() -> None:
    voice = runtime(llm=ollama_missing)

    async def scenario() -> None:
        assert voice.unavailable() == list(VOICE_CHECKS), "everything is loading at first"
        await voice.load()

    asyncio.run(scenario())
    assert voice.unavailable() == ["llm"]
    assert voice.state("stt") == "ok"


def test_disabled_runtime_is_unavailable_and_loads_nothing() -> None:
    called: list[str] = []

    async def spy() -> object:
        called.append("loaded")
        return object()

    voice = VoiceRuntime(enabled=False, loaders=dict.fromkeys(VOICE_CHECKS, spy))
    asyncio.run(voice.load())

    assert called == []
    assert voice.unavailable() == list(VOICE_CHECKS)
    assert {voice.state(name) for name in VOICE_CHECKS} == {"disabled"}


def test_runtime_needs_a_loader_per_voice_check() -> None:
    with pytest.raises(ValueError, match="vad"):
        VoiceRuntime(enabled=True, loaders={"stt": loaded, "tts": loaded, "llm": loaded})


def test_components_are_reachable_only_once_loaded() -> None:
    voice = runtime()

    with pytest.raises(LookupError, match="stt"):
        voice.component("stt")
    asyncio.run(voice.load())
    assert voice.component("stt") is not None


def test_closing_the_runtime_closes_what_was_loaded() -> None:
    closed: list[str] = []

    class Closable:
        async def aclose(self) -> None:
            closed.append("llm")

    async def closable() -> object:
        return Closable()

    voice = runtime(llm=closable)

    async def scenario() -> None:
        await voice.load()
        await voice.aclose()

    asyncio.run(scenario())
    assert closed == ["llm"]
