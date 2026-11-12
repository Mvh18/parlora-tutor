"""Application factory (design D8).

Importing this module has no side effects: nothing reads the environment,
connects to the database or loads a model until `create_app` runs. Serve it with
`uvicorn --factory parlora_server.main:create_app`.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from parlora_core.languages import default_registry
from parlora_server import __version__
from parlora_server.api.middleware import (
    REQUEST_ID_HEADER,
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
)
from parlora_server.api.problems import PROBLEM_JSON, Problem, install_problem_handlers
from parlora_server.api.v1 import calls, catalog, health
from parlora_server.config import Settings, load_settings
from parlora_server.db import Database
from parlora_server.logs import configure_logging
from parlora_server.voice.registry import CallRegistry, CallStarter
from parlora_server.voice.runtime import VOICE_CHECKS, VoiceRuntime

API_PREFIX = "/api/v1"
TITLE = "Parlora API"
DESCRIPTION = (
    "Versioned HTTP API of Parlora, a local-first voice tutor. "
    "Every error is an RFC 9457 problem document (`application/problem+json`)."
)
# Every request body is JSON of a few kilobytes at most (an SDP offer is the largest).
MAX_BODY_BYTES = 64 * 1024


def create_app(
    settings: Settings | None = None,
    *,
    voice: VoiceRuntime | None = None,
    call_starter: CallStarter | None = None,
) -> FastAPI:
    """Build the application; without `settings`, load them from the environment.

    `voice` and `call_starter` replace the real models and WebRTC calls in tests.
    """
    settings = settings if settings is not None else load_settings()
    configure_logging(settings.log_level)
    database = Database(settings.database)
    voice = voice if voice is not None else VoiceRuntime.from_settings(settings)
    registry = CallRegistry(
        max_calls=settings.voice.max_calls,
        max_call_seconds=settings.voice.max_call_seconds,
        idle_seconds=settings.voice.idle_seconds,
    )
    if call_starter is None:
        from parlora_server.voice.call import WebRTCCallStarter

        call_starter = WebRTCCallStarter(voice, settings)

    async def database_state() -> str:
        return "ok" if await database.ping() else "failed"

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        # Models load in the background: liveness answers at once, readiness
        # reports `loading` until they are warmed up.
        background = [
            asyncio.create_task(voice.load(), name="voice loading"),
            asyncio.create_task(registry.run_watchdog(), name="call watchdog"),
        ]
        try:
            yield
        finally:
            for task in background:
                task.cancel()
            for task in background:
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            await registry.close_all()
            await voice.aclose()
            await database.dispose()

    app = FastAPI(
        title=TITLE,
        version=__version__,
        description=DESCRIPTION,
        openapi_url=f"{API_PREFIX}/openapi.json",
        docs_url=f"{API_PREFIX}/docs",
        redoc_url=None,
        generate_unique_id_function=_operation_id,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.database = database
    app.state.voice = voice
    app.state.calls = registry
    app.state.call_starter = call_starter
    app.state.readiness_checks = {
        "database": database_state,
        **{name: voice.check(name) for name in VOICE_CHECKS},
    }
    app.state.language_registry = default_registry()

    install_problem_handlers(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", REQUEST_ID_HEADER],
        expose_headers=[REQUEST_ID_HEADER],
    )
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=MAX_BODY_BYTES)
    # Added last, so it wraps CORS and every response gets the request id.
    app.add_middleware(RequestContextMiddleware)

    v1 = APIRouter(prefix=API_PREFIX)
    v1.include_router(health.router)
    v1.include_router(catalog.router)
    v1.include_router(calls.router)
    app.include_router(v1)

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = _openapi_with_problems(app)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]
    return app


def _openapi_with_problems(app: FastAPI) -> dict[str, Any]:
    """The generated schema plus the problem document as every operation's default response."""
    schema = get_openapi(
        title=app.title, version=app.version, description=app.description, routes=app.routes
    )
    schemas = schema.setdefault("components", {}).setdefault("schemas", {})
    schemas["Problem"] = Problem.model_json_schema(ref_template="#/components/schemas/{model}")
    reference = {"$ref": "#/components/schemas/Problem"}
    for operations in schema["paths"].values():
        for operation in operations.values():
            responses = operation.setdefault("responses", {})
            for status, response in responses.items():
                if not status.startswith("2"):
                    response["content"] = {PROBLEM_JSON: {"schema": reference}}
            responses["default"] = {
                "description": "Problem document (RFC 9457)",
                "content": {PROBLEM_JSON: {"schema": reference}},
            }
    return schema


def _operation_id(route: APIRoute) -> str:
    """Stable, readable ids for generated clients, e.g. `health_live`."""
    return f"{route.tags[0]}_{route.name}" if route.tags else route.name
