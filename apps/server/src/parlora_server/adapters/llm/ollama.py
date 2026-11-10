"""Streaming language-model adapter on Ollama's native chat API.

The native `/api/chat` endpoint is used instead of the OpenAI-compatible `/v1`
one, so the model's load and evaluation timings stay visible. Each reply is
streamed (`stream: true`) and its text yielded chunk by chunk as Ollama writes it;
thinking text is never yielded, and `think: false` is sent to models that declare
the `thinking` capability (asked once per adapter through `/api/show`). Every
request passes `keep_alive` (by default -1: the model stays resident between
turns) and `num_ctx`.

Every failure (Ollama unreachable, an HTTP error, an error inside the stream, a
malformed or cut stream) becomes a `LanguageModelError`. Error messages name the
endpoint, the model and Ollama's own error, never the conversation.
"""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx2

from parlora_core.errors import LanguageModelError
from parlora_core.ports import Message
from parlora_server.config import LlmSettings

# A cold model can take several seconds to load before the first chunk arrives.
TIMEOUT = httpx2.Timeout(120.0, connect=5.0)


def _error_text(response_body: bytes) -> str:
    try:
        decoded = json.loads(response_body)
    except ValueError:
        return "no error message"
    if isinstance(decoded, Mapping) and isinstance(decoded.get("error"), str):
        return str(decoded["error"])
    return "no error message"


class OllamaLanguageModel:
    """`parlora_core.ports.LanguageModel` over Ollama's native streaming chat API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        num_ctx: int = 4096,
        keep_alive: int = -1,
        client: httpx2.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._client = client or httpx2.AsyncClient(timeout=TIMEOUT)
        self._capabilities: frozenset[str] | None = None
        self._capabilities_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def capabilities(self) -> frozenset[str]:
        """The model's capabilities from `/api/show`, asked once."""
        async with self._capabilities_lock:
            if self._capabilities is None:
                response = await self._send("POST", "/api/show", {"model": self.model})
                found = response.json().get("capabilities") or []
                self._capabilities = frozenset(str(c) for c in found)
            return self._capabilities

    async def stream_reply(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"num_ctx": self.num_ctx},
        }
        if "thinking" in await self.capabilities():
            body["think"] = False
        try:
            async with self._client.stream(
                "POST", self.base_url + "/api/chat", json=body
            ) as response:
                if response.status_code != 200:
                    await response.aread()
                    raise self._http_error(response)
                done = False
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = self._decode(line)
                    text = chunk.get("message", {}).get("content") or ""
                    if text:
                        yield str(text)
                    if chunk.get("done"):
                        done = True
                        break
        except httpx2.HTTPError as error:
            raise self._unreachable(error) from error
        if not done:
            raise LanguageModelError(f"{self.model}: the reply stream ended before it was done")

    async def _send(self, method: str, path: str, body: Mapping[str, Any]) -> httpx2.Response:
        try:
            response = await self._client.request(method, self.base_url + path, json=body)
        except httpx2.HTTPError as error:
            raise self._unreachable(error) from error
        if response.status_code != 200:
            raise self._http_error(response)
        return response

    def _decode(self, line: str) -> Mapping[str, Any]:
        try:
            chunk = json.loads(line)
        except ValueError:
            raise LanguageModelError(
                f"{self.model}: Ollama sent a chunk that is not JSON"
            ) from None
        if not isinstance(chunk, Mapping):
            raise LanguageModelError(f"{self.model}: Ollama sent a chunk that is not an object")
        if "error" in chunk:
            raise LanguageModelError(f"{self.model}: Ollama error: {chunk['error']}")
        return chunk

    def _http_error(self, response: httpx2.Response) -> LanguageModelError:
        return LanguageModelError(
            f"{self.model}: Ollama answered HTTP {response.status_code}: "
            f"{_error_text(response.content)}"
        )

    def _unreachable(self, error: httpx2.HTTPError) -> LanguageModelError:
        return LanguageModelError(
            f"Ollama is not answering on {self.base_url}: {type(error).__name__}"
        )


def load_ollama(settings: LlmSettings) -> OllamaLanguageModel:
    """The adapter the settings describe; nothing is sent until the first reply."""
    return OllamaLanguageModel(
        str(settings.base_url),
        settings.model,
        num_ctx=settings.num_ctx,
        keep_alive=settings.keep_alive,
    )
