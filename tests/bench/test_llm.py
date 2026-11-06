"""The LLM benchmark against a fake Ollama server: no network, no GPU."""

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts.bench import llm
from scripts.bench.common.schema import validate_result

pytestmark = pytest.mark.unit

INSTALLED = ["qwen3:4b", "gemma3:4b"]


class FakeOllama(BaseHTTPRequestHandler):
    bodies: list[dict[str, Any]]
    loaded: set[str]

    def log_message(self, format: str, *args: Any) -> None:  # silence the test output
        pass

    def _json(self, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/api/version":
            self._json({"version": "0.40.15"})
        elif self.path == "/api/tags":
            models = [
                {"name": m, "digest": "d" * 64, "size": 1 << 30, "details": {}} for m in INSTALLED
            ]
            self._json({"models": models})
        elif self.path == "/api/ps":
            models = [{"name": m, "size_vram": 3 << 30} for m in sorted(self.loaded)]
            self._json({"models": models})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.bodies.append(body)
        if self.path == "/api/show":
            thinking = ["thinking"] if body["model"].startswith("qwen3") else []
            self._json({"capabilities": ["completion", *thinking]})
            return
        if self.path == "/api/generate" and body.get("keep_alive") == 0:
            self.loaded.discard(body["model"])
            self._json({"done": True})
            return
        self.loaded.add(body["model"])
        chunks = [
            {"message": {"content": ""}, "done": False},
            {"message": {"content": "Hello"}, "done": False},
            {
                "message": {"content": ""},
                "done": True,
                "done_reason": "length",
                "total_duration": 900_000_000,
                "load_duration": 500_000_000,
                "prompt_eval_count": 80,
                "prompt_eval_duration": 100_000_000,
                "eval_count": 128,
                "eval_duration": 1_000_000_000,
            },
        ]
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(json.dumps(chunk).encode() + b"\n")
            self.wfile.flush()


class FakeSampler:
    source = "fake"
    peak_mb: int | None = None

    def used_mb(self) -> int:
        return 500

    def __enter__(self) -> "FakeSampler":
        self.peak_mb = 4_100
        return self

    def __exit__(self, *exc: object) -> None:
        pass


@pytest.fixture
def ollama() -> Iterator[tuple[str, type[FakeOllama]]]:
    handler = type("Handler", (FakeOllama,), {"bodies": [], "loaded": set()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", handler
    server.shutdown()
    server.server_close()


def run(endpoint: str, out: Path, *models: str, runs: int = 3) -> int:
    argv = [*models, "--runs", str(runs), "--endpoint", endpoint, "--output-dir", str(out)]
    return llm.main(argv, sampler=FakeSampler())  # type: ignore[arg-type]


def test_one_valid_result_per_model(ollama: tuple[str, type[FakeOllama]], tmp_path: Path) -> None:
    endpoint, _ = ollama

    assert run(endpoint, tmp_path, "qwen3:4b", "gemma3:4b") == 0

    files = sorted(tmp_path.glob("*.json"))
    assert [f.name.split("-", 1)[1] for f in files] == ["llm-gemma3-4b.json", "llm-qwen3-4b.json"]
    result = json.loads(files[1].read_text(encoding="utf-8"))
    validate_result(result)
    measurements = result["measurements"]
    assert measurements["ttft_s"]["summary"]["count"] == 3
    assert measurements["tokens_per_s"]["summary"]["p50"] == 128.0
    assert measurements["load_s"] == {"cold": 0.5}
    assert measurements["vram_mb"] == {
        "baseline": 500,
        "peak": 4_100,
        "peak_above_baseline": 3_600,
        "ollama_size_vram": 3_072,
    }
    assert result["software"]["ollama"] == "0.40.15"
    assert result["checks"][0]["name"] == "llm_warm_ttft"
    assert result["checks"][0]["status"] == "pass"


def test_model_is_unloaded_first_and_options_are_fixed(
    ollama: tuple[str, type[FakeOllama]], tmp_path: Path
) -> None:
    endpoint, handler = ollama

    run(endpoint, tmp_path, "qwen3:4b", runs=2)

    show, unload, *chats = handler.bodies
    assert show == {"model": "qwen3:4b", "stream": False}
    assert unload == {"model": "qwen3:4b", "keep_alive": 0, "stream": False}
    assert len(chats) == 3
    assert all(c["options"] == llm.OPTIONS and c["stream"] is True for c in chats)
    assert all(c["think"] is False for c in chats)
    prompts = [c["messages"][1]["content"] for c in chats]
    assert len(set(prompts)) == 3, "every run sends a different learner prompt"


def test_thinking_is_not_sent_to_models_without_it(
    ollama: tuple[str, type[FakeOllama]], tmp_path: Path
) -> None:
    endpoint, handler = ollama

    run(endpoint, tmp_path, "gemma3:4b", runs=1)

    assert all("think" not in b for b in handler.bodies)


def test_capabilities_are_recorded(ollama: tuple[str, type[FakeOllama]], tmp_path: Path) -> None:
    endpoint, _ = ollama

    run(endpoint, tmp_path, "qwen3:4b", runs=1)

    (path,) = tmp_path.glob("*.json")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["software"]["capabilities"] == ["completion", "thinking"]
    assert result["settings"]["think"] is False


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_ollama_not_running_exits_without_a_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(f"http://127.0.0.1:{free_port()}", tmp_path, "qwen3:4b") == 2

    assert list(tmp_path.iterdir()) == []
    assert "Start Ollama" in capsys.readouterr().err


def test_missing_model_exits_before_any_run(
    ollama: tuple[str, type[FakeOllama]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    endpoint, handler = ollama

    assert run(endpoint, tmp_path, "qwen3:4b", "qwen3:8b") == 2

    assert handler.bodies == []
    assert list(tmp_path.iterdir()) == []
    assert "ollama pull qwen3:8b" in capsys.readouterr().err


def test_runs_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        llm.main(["--runs", "0", "--output-dir", str(tmp_path)])


def test_only_http_endpoints_are_accepted() -> None:
    with pytest.raises(ValueError, match="http"):
        llm.Ollama("file:///etc/passwd")
