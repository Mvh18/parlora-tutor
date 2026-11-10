"""The streaming Ollama language-model adapter against recorded `/api/chat` streams."""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest

from parlora_core.errors import LanguageModelError
from parlora_core.ports import LanguageModel, Message
from parlora_server.adapters.llm import ollama
from parlora_server.config import LlmSettings

pytestmark = pytest.mark.unit

BASE_URL = "http://ollama.test:11434"
MODEL = "qwen3:4b-instruct-2507-q4_K_M"
# Recorded from Ollama 0.40.15 with this model (temperature 0, seed 42).
RECORDED = (Path(__file__).parent / "fixtures" / "ollama" / "chat-stream.ndjson").read_bytes()
MESSAGES = (
    Message("system", "You are a friendly English conversation partner."),
    Message("user", "Yesterday I go to the market and buy some vegetables."),
)

Handler = Callable[[httpx2.Request], httpx2.Response]


class FakeOllama:
    """Answers `/api/show` with `capabilities` and `/api/chat` with `chat`; records requests."""

    def __init__(
        self,
        capabilities: Sequence[str] = ("completion",),
        chat: bytes = RECORDED,
        chat_status: int = 200,
    ) -> None:
        self.capabilities = list(capabilities)
        self.chat = chat
        self.chat_status = chat_status
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content) if request.content else {}
        self.requests.append((request.url.path, body))
        if request.url.path == "/api/show":
            return httpx2.Response(200, json={"capabilities": self.capabilities})
        if request.url.path == "/api/chat":
            return httpx2.Response(self.chat_status, content=self.chat)
        return httpx2.Response(404, json={"error": "not found"})

    def chat_bodies(self) -> list[dict[str, Any]]:
        return [body for path, body in self.requests if path == "/api/chat"]


def adapter(handler: Handler, **options: Any) -> ollama.OllamaLanguageModel:
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    return ollama.OllamaLanguageModel(BASE_URL, MODEL, client=client, **options)


def reply(model: ollama.OllamaLanguageModel, messages: Sequence[Message] = MESSAGES) -> list[str]:
    async def collect() -> list[str]:
        return [chunk async for chunk in model.stream_reply(messages)]

    return asyncio.run(collect())


def test_implements_the_port() -> None:
    model: LanguageModel = adapter(FakeOllama())
    assert model is not None


def test_yields_the_recorded_chunks_in_order_without_empty_ones() -> None:
    chunks = reply(adapter(FakeOllama()))

    assert chunks[:3] == ["That", " sounds", " like"]
    assert "" not in chunks
    assert (
        "".join(chunks)
        == "That sounds like a great way to start the day! What kind of vegetables did you get?"
    )


def test_request_uses_the_native_chat_api_with_resident_model_options() -> None:
    fake = FakeOllama()

    reply(adapter(fake, num_ctx=4096, keep_alive=-1))

    (body,) = fake.chat_bodies()
    assert body["model"] == MODEL
    assert body["stream"] is True
    assert body["keep_alive"] == -1
    assert body["options"] == {"num_ctx": 4096}
    assert body["messages"] == [
        {"role": "system", "content": "You are a friendly English conversation partner."},
        {"role": "user", "content": "Yesterday I go to the market and buy some vegetables."},
    ]


def test_think_false_is_sent_only_to_thinking_models() -> None:
    thinking, plain = FakeOllama(("completion", "thinking")), FakeOllama(("completion",))

    reply(adapter(thinking))
    reply(adapter(plain))

    assert thinking.chat_bodies()[0]["think"] is False
    assert "think" not in plain.chat_bodies()[0]


def test_capabilities_are_asked_once_per_adapter() -> None:
    fake = FakeOllama(("completion", "thinking"))
    model = adapter(fake)

    async def twice() -> None:
        for _ in range(2):
            async for _chunk in model.stream_reply(MESSAGES):
                pass

    asyncio.run(twice())

    assert [path for path, _ in fake.requests].count("/api/show") == 1
    assert len(fake.chat_bodies()) == 2


def test_thinking_text_is_never_yielded() -> None:
    stream = b"\n".join(
        [
            b'{"message":{"role":"assistant","content":"","thinking":"hmm"},"done":false}',
            b'{"message":{"role":"assistant","content":"Hi!"},"done":false}',
            b'{"message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}',
        ]
    )

    assert reply(adapter(FakeOllama(chat=stream))) == ["Hi!"]


def test_an_error_inside_the_stream_is_a_domain_error() -> None:
    stream = (
        b'{"message":{"role":"assistant","content":"Hi"},"done":false}\n{"error":"out of memory"}\n'
    )

    with pytest.raises(LanguageModelError, match="out of memory"):
        reply(adapter(FakeOllama(chat=stream)))


def test_an_http_error_is_a_domain_error_naming_the_status() -> None:
    fake = FakeOllama(chat=b'{"error":"model \'qwen3:4b\' not found"}', chat_status=404)

    with pytest.raises(LanguageModelError, match=r"404.*not found"):
        reply(adapter(fake))


def test_a_missing_model_fails_when_its_capabilities_are_asked() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"error": f"model '{MODEL}' not found"})

    with pytest.raises(LanguageModelError, match="not found"):
        reply(adapter(handler))


def test_an_unreachable_ollama_is_a_domain_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    with pytest.raises(LanguageModelError, match="not answering"):
        reply(adapter(handler))


def test_a_stream_cut_before_done_is_a_domain_error() -> None:
    stream = b'{"message":{"role":"assistant","content":"Hi"},"done":false}\n'

    with pytest.raises(LanguageModelError, match="ended before"):
        reply(adapter(FakeOllama(chat=stream)))


def test_a_malformed_chunk_is_a_domain_error() -> None:
    with pytest.raises(LanguageModelError, match="not JSON"):
        reply(adapter(FakeOllama(chat=b"<html>proxy error</html>\n")))


def test_errors_never_carry_the_conversation() -> None:
    stream = b'{"error":"bad request"}\n'

    with pytest.raises(LanguageModelError) as excinfo:
        reply(adapter(FakeOllama(chat=stream)))

    assert "vegetables" not in str(excinfo.value)


def test_load_builds_the_adapter_from_the_settings() -> None:
    settings = LlmSettings(
        base_url="http://127.0.0.1:11434/", model="gemma3:4b", num_ctx=2048, keep_alive=600
    )

    model = ollama.load_ollama(settings)

    assert (model.base_url, model.model, model.num_ctx, model.keep_alive) == (
        "http://127.0.0.1:11434",
        "gemma3:4b",
        2048,
        600,
    )
    asyncio.run(model.aclose())
