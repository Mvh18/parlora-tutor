"""Liveness and readiness (design D8; `checks` is an open map — D19)."""

import pytest
from fastapi.testclient import TestClient

from parlora_server import __version__
from parlora_server.config import Settings
from parlora_server.main import create_app

pytestmark = pytest.mark.unit


VOICE_DISABLED = {"stt": "disabled", "tts": "disabled", "llm": "disabled", "vad": "disabled"}


async def healthy() -> str:
    return "ok"


async def failing() -> str:
    return "failed"


def test_live_reports_ok_and_the_version(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_ready_when_every_check_passes(settings: Settings) -> None:
    app = create_app(settings)
    app.state.readiness_checks["database"] = healthy

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "checks": {"database": "ok", **VOICE_DISABLED}}


def test_not_ready_is_a_503_problem_listing_the_failed_check(settings: Settings) -> None:
    app = create_app(settings)
    app.state.readiness_checks = {"database": failing, "models": healthy}

    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/problem+json"
    body = response.json()
    assert body["status"] == 503
    assert body["instance"] == "/api/v1/health/ready"
    assert body["checks"] == {"database": "failed", "models": "ok"}


def test_ready_without_a_database_hides_the_connection_string(client: TestClient) -> None:
    # The default checks ping the configured database, which is unreachable here.
    response = client.get("/api/v1/health/ready")

    assert response.status_code == 503
    assert response.json()["checks"] == {"database": "failed", **VOICE_DISABLED}
    assert "postgresql" not in response.text
    assert "127.0.0.1:1" not in response.text
