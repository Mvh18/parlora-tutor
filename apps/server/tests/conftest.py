"""Fixtures shared by the server's unit, integration and contract suites."""

import os
import sys
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from testcontainers.community.postgres import PostgresContainer
from testcontainers.core.docker_client import DockerClient

from parlora_server.config import Settings
from parlora_server.main import create_app

# A database URL nothing listens on: unit tests never reach PostgreSQL.
UNREACHABLE_DATABASE_URL = "postgresql+asyncpg://parlora:parlora@127.0.0.1:1/parlora"


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every PARLORA_ variable so the developer's environment cannot leak in."""
    for name in list(os.environ):
        if name.upper().startswith("PARLORA_"):
            monkeypatch.delenv(name)


@pytest.fixture
def make_settings(clean_env: None) -> Callable[..., Settings]:
    def make(**overrides: object) -> Settings:
        # Voice is off unless a test turns it on: unit tests never load a model.
        values: dict[str, object] = {
            "database": {"url": UNREACHABLE_DATABASE_URL},
            "voice": {"enabled": False},
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)  # type: ignore[arg-type]

    return make


@pytest.fixture
def settings(make_settings: Callable[..., Settings]) -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    # Server errors become 500 responses, as they do behind uvicorn.
    with TestClient(create_app(settings), raise_server_exceptions=False) as test_client:
        yield test_client


# ── Integration: PostgreSQL in a throwaway container ────────────────────────

POSTGRES_IMAGE = "postgres:18-alpine"


def docker_available() -> bool:
    try:
        DockerClient().client.ping()
    except Exception:  # no daemon, no socket, no permission
        return False
    return True


def start_postgres() -> PostgresContainer:
    if not docker_available():
        if os.environ.get("CI"):
            pytest.fail("Docker is required for the integration tests in CI")
        pytest.skip("Docker is not available")
    if sys.platform == "win32":
        # Docker Desktop publishes ports a moment after start, and testcontainers
        # reads Ryuk's port once without retrying. The fixtures stop every
        # container themselves; only a killed run can leave one behind.
        os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")
    return PostgresContainer(POSTGRES_IMAGE, driver=None).start()


def asyncpg_url(container: PostgresContainer) -> str:
    host = container.get_container_host_ip().replace("localhost", "127.0.0.1")
    port = container.get_exposed_port(5432)
    return (
        f"postgresql+asyncpg://{container.username}:{container.password}"
        f"@{host}:{port}/{container.dbname}"
    )


@pytest.fixture(scope="session")
def postgres() -> Iterator[PostgresContainer]:
    """One PostgreSQL for the whole session; tests must not stop it."""
    container = start_postgres()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def disposable_postgres() -> Iterator[PostgresContainer]:
    """A PostgreSQL of its own, for tests that stop it."""
    container = start_postgres()
    try:
        yield container
    finally:
        container.stop()


@pytest.fixture
def postgres_url(postgres: PostgresContainer) -> str:
    return asyncpg_url(postgres)
