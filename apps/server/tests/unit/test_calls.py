"""Call signalling under /api/v1/calls with a fake call starter (design D6).

The fake starter records what reached it, so "no peer connection is created" is
checked as "the starter was never called".
"""

import asyncio
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi.testclient import TestClient

from parlora_core.cefr import CefrLevel
from parlora_core.errors import LanguageModelError
from parlora_core.session import SessionSettings
from parlora_server.config import Settings
from parlora_server.main import create_app
from parlora_server.voice.registry import (
    CallHooks,
    IceCandidate,
    InvalidOfferError,
    SdpDescription,
)
from parlora_server.voice.runtime import VOICE_CHECKS, VoiceRuntime

pytestmark = pytest.mark.unit

SDP = "v=0\r\no=- 4611 2 IN IP4 127.0.0.1\r\ns=SECRET-SDP-MARKER\r\n"
ANSWER = SdpDescription("v=0\r\ns=answer\r\n", "answer")
SETTINGS = {"target_language": "en", "level": "B1", "topic": "Weekend plans"}
OFFER = "/api/v1/calls/offer"


@dataclass
class FakeSession:
    closed: int = 0
    candidates: list[IceCandidate] = field(default_factory=list)

    async def add_ice_candidates(self, candidates: Sequence[IceCandidate]) -> None:
        self.candidates.extend(candidates)

    async def close(self) -> None:
        self.closed += 1


@dataclass
class FakeStarter:
    error: Exception | None = None
    offers: list[tuple[SdpDescription, SessionSettings]] = field(default_factory=list)
    sessions: list[FakeSession] = field(default_factory=list)
    hooks: list[CallHooks] = field(default_factory=list)

    async def __call__(
        self, offer: SdpDescription, settings: SessionSettings, hooks: CallHooks
    ) -> tuple[FakeSession, SdpDescription]:
        self.offers.append((offer, settings))
        if self.error is not None:
            raise self.error
        session = FakeSession()
        self.sessions.append(session)
        self.hooks.append(hooks)
        return session, ANSWER


async def loaded() -> object:
    return object()


async def ollama_missing() -> object:
    raise LanguageModelError("Ollama answered HTTP 404")


def loaded_runtime(**overrides: Callable[[], Any]) -> VoiceRuntime:
    loaders: dict[str, Callable[[], Any]] = dict.fromkeys(VOICE_CHECKS, loaded)
    loaders.update(overrides)
    runtime = VoiceRuntime(enabled=True, loaders=loaders)
    asyncio.run(runtime.load())
    return runtime


@pytest.fixture
def starter() -> FakeStarter:
    return FakeStarter()


def calls_client(
    settings: Settings, starter: FakeStarter, voice: VoiceRuntime | None = None
) -> TestClient:
    app = create_app(settings, voice=voice or loaded_runtime(), call_starter=starter)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def client(settings: Settings, starter: FakeStarter) -> Iterator[TestClient]:
    with calls_client(settings, starter) as test_client:
        yield test_client


def offer(client: TestClient, **settings: Any) -> Any:
    return client.post(
        OFFER, json={"sdp": SDP, "type": "offer", "settings": {**SETTINGS, **settings}}
    )


def assert_problem(response: Any, status: int, type_: str = "about:blank") -> dict[str, Any]:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    body: dict[str, Any] = response.json()
    assert body["type"] == type_
    assert "SECRET-SDP-MARKER" not in response.text, "problems never echo the SDP"
    return body


# ── Offer ───────────────────────────────────────────────────────────────────


def test_an_offer_starts_a_call_with_validated_settings(
    client: TestClient, starter: FakeStarter
) -> None:
    response = offer(client, topic="  Weekend plans  ")

    assert response.status_code == 200
    body = response.json()
    assert body == {"call_id": body["call_id"], "sdp": ANSWER.sdp, "type": "answer"}
    ((sdp, settings),) = starter.offers
    assert sdp == SdpDescription(SDP, "offer")
    assert settings.level is CefrLevel.B1
    assert str(settings.target_language) == "en"
    assert str(settings.feedback_language) == "en", "feedback defaults to the target language"
    assert settings.topic == "Weekend plans"
    assert str(settings.learner_ref).startswith("ephemeral:"), "nothing identifies the learner"


@pytest.mark.parametrize(
    "changes",
    [
        {"level": "Z9"},
        {"target_language": "xx"},
        {"target_language": "EN"},
        {"feedback_language": "zz"},
        {"topic": "   "},
        {"topic": "x" * 121},
    ],
)
def test_invalid_settings_are_422_and_create_no_peer_connection(
    client: TestClient, starter: FakeStarter, changes: dict[str, str]
) -> None:
    body = assert_problem(offer(client, **changes), 422)

    assert body["errors"]
    assert starter.offers == []


def test_an_offer_must_be_an_sdp_offer(client: TestClient, starter: FakeStarter) -> None:
    response = client.post(OFFER, json={"sdp": SDP, "type": "answer", "settings": SETTINGS})

    assert_problem(response, 422)
    assert starter.offers == []


def test_an_offer_the_peer_connection_rejects_is_400(
    client: TestClient, starter: FakeStarter
) -> None:
    starter.error = InvalidOfferError("aiortc: invalid SDP line SECRET-SDP-MARKER")

    body = assert_problem(offer(client), 400)
    assert body["detail"] == "The SDP offer could not be used."

    starter.error = None
    assert offer(client).status_code == 200, "the failed offer freed its slot"


def test_the_call_limit_is_409(client: TestClient, starter: FakeStarter) -> None:
    assert offer(client).status_code == 200

    body = assert_problem(offer(client), 409, "/problems/call-limit")

    assert body["max_calls"] == 1
    assert len(starter.offers) == 1


def test_voice_unavailable_is_503_naming_the_dependency(
    settings: Settings, starter: FakeStarter
) -> None:
    with calls_client(settings, starter, loaded_runtime(llm=ollama_missing)) as client:
        body = assert_problem(offer(client), 503, "/problems/voice-unavailable")

    assert body["unavailable"] == ["llm"]
    assert "llm" in body["detail"]
    assert starter.offers == []


def test_voice_disabled_is_503_naming_every_dependency(
    settings: Settings, starter: FakeStarter
) -> None:
    disabled = VoiceRuntime(enabled=False, loaders={})
    with calls_client(settings, starter, disabled) as client:
        body = assert_problem(offer(client), 503, "/problems/voice-unavailable")

    assert body["unavailable"] == list(VOICE_CHECKS)
    assert starter.offers == []


def test_an_oversized_request_is_413(client: TestClient, starter: FakeStarter) -> None:
    response = client.post(
        OFFER,
        json={"sdp": "a=x\r\n" * 20_000, "type": "offer", "settings": SETTINGS},
    )

    assert_problem(response, 413)
    assert starter.offers == []


# ── ICE candidates and end ──────────────────────────────────────────────────


def test_ice_candidates_reach_the_call(client: TestClient, starter: FakeStarter) -> None:
    call_id = offer(client).json()["call_id"]
    candidates: list[dict[str, Any]] = [
        {
            "candidate": "candidate:1 1 udp 2122260223 127.0.0.1 50000 typ host",
            "sdp_mid": "0",
            "sdp_mline_index": 0,
        },
        {"candidate": "", "sdp_mid": "0", "sdp_mline_index": 0},
    ]

    response = client.patch(f"/api/v1/calls/{call_id}", json={"candidates": candidates})

    assert response.status_code == 204
    assert starter.sessions[0].candidates == [IceCandidate(**c) for c in candidates]


def test_ending_a_call_closes_it_and_frees_the_slot(
    client: TestClient, starter: FakeStarter
) -> None:
    call_id = offer(client).json()["call_id"]

    assert client.delete(f"/api/v1/calls/{call_id}").status_code == 204
    assert starter.sessions[0].closed == 1
    assert offer(client).status_code == 200


def test_a_call_that_ends_on_its_own_frees_the_slot(
    client: TestClient, starter: FakeStarter
) -> None:
    call_id = offer(client).json()["call_id"]

    starter.hooks[0].ended()

    assert_problem(client.delete(f"/api/v1/calls/{call_id}"), 404)
    assert offer(client).status_code == 200


@pytest.mark.parametrize("method", ["delete", "patch"])
def test_unknown_calls_are_404(client: TestClient, method: str) -> None:
    kwargs: dict[str, Any] = {"json": {"candidates": []}} if method == "patch" else {}

    response = getattr(client, method)("/api/v1/calls/0123456789abcdef0123456789abcdef", **kwargs)

    body = assert_problem(response, 404)
    assert body["instance"] == "/api/v1/calls/0123456789abcdef0123456789abcdef"
