"""Call signalling: offer, ICE candidates and end (design D6).

The offer carries the call settings, validated by the domain's `SessionSettings`
with an ephemeral learner reference; nothing about the call is stored. Offers are
refused before any peer connection exists when the settings are invalid (422),
a voice dependency is not ready (503, `/problems/voice-unavailable`) or the
server already holds its maximum number of calls (409, `/problems/call-limit`).
Problem documents never echo the SDP.
"""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Path, Request, Response
from pydantic import BaseModel, ConfigDict, Field

from parlora_core.cefr import CefrLevel
from parlora_core.errors import DomainError
from parlora_core.languages import LanguageCode, LanguageRegistry
from parlora_core.learner import LearnerRef
from parlora_core.session import MAX_TOPIC_LENGTH, SessionSettings
from parlora_server.api.problems import ProblemError
from parlora_server.voice.registry import (
    CallHooks,
    CallLimitError,
    CallRegistry,
    CallSession,
    CallStarter,
    IceCandidate,
    InvalidOfferError,
    SdpDescription,
    UnknownCallError,
)
from parlora_server.voice.runtime import VoiceRuntime

CALL_LIMIT = "/problems/call-limit"
VOICE_UNAVAILABLE = "/problems/voice-unavailable"
# A browser's audio-only offer is a few kilobytes.
MAX_SDP_LENGTH = 32_768
MAX_CANDIDATES = 64

router = APIRouter(prefix="/calls", tags=["calls"])

LanguageField = Annotated[str, Field(pattern=r"^[a-z]{2}$", examples=["en"])]
CallId = Annotated[str, Path(max_length=64, description="The id the offer answered with.")]


class CallSettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid", title="CallSettings")

    target_language: LanguageField = Field(description="Language practised in the call.")
    feedback_language: LanguageField | None = Field(
        default=None,
        description="Language corrections are explained in; the target language when omitted.",
    )
    level: CefrLevel = Field(description="CEFR level of the learner.")
    topic: str = Field(
        min_length=1, max_length=MAX_TOPIC_LENGTH, description="What the call is about."
    )


class CallOffer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sdp: str = Field(min_length=1, max_length=MAX_SDP_LENGTH, description="SDP offer.")
    type: Literal["offer"]
    settings: CallSettingsIn


class CallAnswer(BaseModel):
    call_id: str = Field(description="Id of the call for ICE candidates and ending it.")
    sdp: str = Field(description="SDP answer.")
    type: Literal["answer"]


class IceCandidateIn(BaseModel):
    model_config = ConfigDict(extra="forbid", title="IceCandidate")

    candidate: str = Field(
        max_length=1024, description="Candidate line; empty marks the end of candidates."
    )
    sdp_mid: str | None = Field(default=None, max_length=64)
    sdp_mline_index: int | None = Field(default=None, ge=0, le=64)


class IceCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates: list[IceCandidateIn] = Field(max_length=MAX_CANDIDATES)


def _invalid(field: str, message: str) -> ProblemError:
    return ProblemError(
        422,
        "The request is not valid.",
        extensions={"errors": [{"loc": ["body", "settings", field], "msg": message}]},
    )


def session_settings(body: CallSettingsIn, languages: LanguageRegistry) -> SessionSettings:
    """The domain's settings for a call; messages name fields, never the values sent."""
    try:
        target = LanguageCode(body.target_language)
        languages.require_target(target)
    except DomainError:
        raise _invalid("target_language", "is not a supported target language") from None
    try:
        feedback = LanguageCode(body.feedback_language or body.target_language)
        languages.require_feedback(feedback)
    except DomainError:
        raise _invalid("feedback_language", "is not a supported feedback language") from None
    try:
        return SessionSettings(
            # Calls are not stored yet (Phase 5): the learner stays anonymous.
            learner_ref=LearnerRef(f"ephemeral:{uuid.uuid4().hex}"),
            target_language=target,
            feedback_language=feedback,
            level=body.level,
            topic=body.topic,
            registry=languages,
        )
    except DomainError:
        raise _invalid("topic", f"must be 1-{MAX_TOPIC_LENGTH} characters after trimming") from None


@router.post(
    "/offer",
    summary="Start a call from a WebRTC offer",
    responses={
        400: {"description": "The SDP offer could not be used."},
        409: {"description": f"The server holds its maximum number of calls (`{CALL_LIMIT}`)."},
        413: {"description": "The request body is too large."},
        503: {
            "description": (
                f"A voice dependency is not ready (`{VOICE_UNAVAILABLE}`); `unavailable` "
                "lists which."
            )
        },
    },
)
async def offer(request: Request, body: CallOffer) -> CallAnswer:
    state = request.app.state
    settings = session_settings(body.settings, state.language_registry)
    voice: VoiceRuntime = state.voice
    unavailable = voice.unavailable()
    if unavailable:
        raise ProblemError(
            503,
            f"Calls are unavailable: {', '.join(unavailable)} not ready.",
            extensions={
                "unavailable": unavailable,
                "checks": {name: voice.state(name) for name in unavailable},
            },
            type=VOICE_UNAVAILABLE,
        )
    calls: CallRegistry = state.calls
    start_call: CallStarter = state.call_starter

    async def start(call_id: str) -> tuple[CallSession, SdpDescription]:
        hooks = CallHooks(
            activity=lambda: calls.touch(call_id), ended=lambda: calls.discard(call_id)
        )
        return await start_call(SdpDescription(body.sdp, body.type), settings, hooks)

    try:
        call_id, answer = await calls.open(start)
    except CallLimitError as error:
        raise ProblemError(
            409,
            "The server already holds its maximum number of calls.",
            extensions={"max_calls": error.max_calls},
            type=CALL_LIMIT,
        ) from None
    except InvalidOfferError:
        raise ProblemError(400, "The SDP offer could not be used.") from None
    return CallAnswer(call_id=call_id, sdp=answer.sdp, type="answer")


@router.patch(
    "/{call_id}",
    summary="Add the client's ICE candidates to a call",
    status_code=204,
    responses={404: {"description": "No active call has this id."}},
)
async def add_ice_candidates(request: Request, call_id: CallId, body: IceCandidates) -> Response:
    calls: CallRegistry = request.app.state.calls
    candidates = [IceCandidate(c.candidate, c.sdp_mid, c.sdp_mline_index) for c in body.candidates]
    try:
        await calls.add_ice_candidates(call_id, candidates)
    except UnknownCallError:
        raise ProblemError(404, "No active call has this id.") from None
    return Response(status_code=204)


@router.delete(
    "/{call_id}",
    summary="End a call",
    status_code=204,
    responses={404: {"description": "No active call has this id."}},
)
async def end(request: Request, call_id: CallId) -> Response:
    calls: CallRegistry = request.app.state.calls
    try:
        await calls.end(call_id)
    except UnknownCallError:
        raise ProblemError(404, "No active call has this id.") from None
    return Response(status_code=204)
