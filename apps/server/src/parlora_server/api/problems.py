"""RFC 9457 problem documents for every error response (design D8).

Problems never carry stack traces, settings or rejected input: a client gets
the status, a short title and detail, the request path and the request id to
quote when reporting the error.
"""

import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from parlora_server.api.middleware import BodyTooLargeError, request_id_of, response_headers

PROBLEM_JSON = "application/problem+json"

logger = logging.getLogger(__name__)


class Problem(BaseModel):
    """An RFC 9457 problem document; extra members carry problem-specific data."""

    model_config = ConfigDict(extra="allow", title="Problem")

    type: str = Field(default="about:blank", description="URI identifying the problem type.")
    title: str = Field(description="Short summary of the problem type.")
    status: int = Field(description="HTTP status code.")
    detail: str = Field(description="Explanation specific to this occurrence.")
    instance: str = Field(description="Path of the request that caused the problem.")
    request_id: str | None = Field(description="Request id to quote when reporting the error.")


class ProblemResponse(JSONResponse):
    media_type = PROBLEM_JSON


class ProblemError(Exception):
    """Raise from a route to answer with a problem document."""

    def __init__(
        self,
        status: int,
        detail: str,
        extensions: Mapping[str, Any] | None = None,
        *,
        type: str = "about:blank",
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.extensions = dict(extensions or {})
        self.type = type


def problem_response(
    request: Request,
    status: int,
    detail: str,
    extensions: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    *,
    type: str = "about:blank",
) -> ProblemResponse:
    problem = Problem(
        type=type,
        title=HTTPStatus(status).phrase,
        status=status,
        detail=detail,
        instance=request.url.path,
        request_id=request_id_of(request),
        **dict(extensions or {}),
    )
    return ProblemResponse(problem.model_dump(), status_code=status, headers=headers)


async def _problem_error(request: Request, exc: Exception) -> ProblemResponse:
    assert isinstance(exc, ProblemError)  # noqa: S101 - registered for ProblemError only
    return problem_response(request, exc.status, exc.detail, exc.extensions, type=exc.type)


async def _http_error(request: Request, exc: Exception) -> ProblemResponse:
    assert isinstance(exc, StarletteHTTPException)  # noqa: S101 - registered for it only
    if isinstance(exc.__cause__, BodyTooLargeError):
        # FastAPI turns any error while reading the body into a 400.
        return await _body_too_large(request, exc.__cause__)
    detail = exc.detail if isinstance(exc.detail, str) else HTTPStatus(exc.status_code).phrase
    return problem_response(request, exc.status_code, detail, headers=exc.headers)


async def _validation_error(request: Request, exc: Exception) -> ProblemResponse:
    assert isinstance(exc, RequestValidationError)  # noqa: S101 - registered for it only
    # Location and message only: the rejected input is not echoed back.
    errors = [{"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()]
    return problem_response(
        request, 422, "The request is not valid.", extensions={"errors": errors}
    )


async def _body_too_large(request: Request, exc: Exception) -> ProblemResponse:
    return problem_response(request, 413, "The request body is too large.")


async def _unhandled_error(request: Request, exc: Exception) -> ProblemResponse:
    request_id = request_id_of(request)
    logger.error("unhandled error (request id %s)", request_id, exc_info=exc)
    # This response bypasses the request middleware, so it sets its headers itself.
    return problem_response(
        request,
        500,
        "An unexpected error occurred. Quote the request id when reporting it.",
        headers=response_headers(request_id) if request_id else None,
    )


def install_problem_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProblemError, _problem_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(BodyTooLargeError, _body_too_large)
    app.add_exception_handler(Exception, _unhandled_error)
