"""LLM benchmark through the host's native Ollama API.

For each model: unload it (`keep_alive: 0`), send one cold chat request, then N
warm requests with fixed tutor-like prompts. Options are fixed (`num_ctx` 4096,
`temperature` 0, `seed`, `num_predict` 128) and thinking is off for models that
support it. Time to first token is measured client-side; Ollama's durations come
from its final chunk (`scripts.bench.common.ollama`). One result per model is
written to `benchmarks/results/`.

Usage: python -m scripts.bench.llm [MODEL ...] [--runs N] [--endpoint URL]

The OpenAI-compatible `/v1` endpoint (what the server will use) is not used here:
it hides Ollama's load, prompt and generation timings.
"""

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.bench.common.budgets import BUDGETS, evaluate_budget
from scripts.bench.common.ollama import (
    StreamTimings,
    model_size_vram_mb,
    model_software,
    parse_stream,
)
from scripts.bench.common.results import RESULTS_DIR, write_result
from scripts.bench.common.schema import build_result, slug
from scripts.bench.common.stats import ColdWarm
from scripts.bench.gpu import VramSampler
from scripts.bench.host import OLLAMA, collect_machine

# `qwen3:4b` now points to Qwen3-4B-Thinking-2507, which always writes its
# reasoning into the reply; the instruct build of the same size does not.
DEFAULT_MODELS = ("qwen3:4b-instruct-2507-q4_K_M", "gemma3:4b", "qwen3:8b")
DEFAULT_RUNS = 10
OPTIONS = {"num_ctx": 4096, "temperature": 0, "seed": 42, "num_predict": 128}
WARM_KEEP_ALIVE = "10m"

SYSTEM_PROMPT = (
    "You are a friendly English conversation partner for a learner at CEFR level B1. "
    "Reply in one or two short spoken sentences, without lists, markdown or emojis, "
    "and end with a simple question that keeps the conversation going."
)
# One cold prompt, then one distinct prompt per warm run (cycled when N > 10), so
# every warm run evaluates new learner text after the cached system prompt.
PROMPTS = (
    "Hello! I want practise my English today.",
    "Yesterday I go to the market and buy some vegetables.",
    "My favourite food is pasta, I cook it every weekends.",
    "I am working in a bank since three years.",
    "Can you explain me the difference between say and tell?",
    "I like travelling but planes make me nervous.",
    "Last summer we have visited my grandparents in the countryside.",
    "What do you think about learning languages with films?",
    "I didn't sleep good last night because of the noise.",
    "My sister is more taller than me.",
    "Tomorrow I will to start a new job, I am excited.",
)


class OllamaError(RuntimeError):
    """Ollama is not answering, or answered with an error."""


class Ollama:
    def __init__(self, endpoint: str = OLLAMA, timeout_s: float = 600) -> None:
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError(f"Ollama endpoint must be http(s), got {endpoint!r}")
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s

    def _open(self, path: str, body: dict[str, Any] | None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(  # noqa: S310 - scheme checked in __init__
            self.endpoint + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout_s)  # noqa: S310
        except urllib.error.HTTPError as error:
            raise OllamaError(f"{path}: HTTP {error.code} {error.read().decode()}") from error
        except (urllib.error.URLError, OSError) as error:
            raise OllamaError(f"Ollama is not answering on {self.endpoint}: {error}") from error

    def get(self, path: str) -> dict[str, Any]:
        with self._open(path, None) as response:
            return dict(json.load(response))

    def post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        with self._open(path, {**body, "stream": False}) as response:
            return dict(json.load(response))

    def stream(self, path: str, body: dict[str, Any]) -> list[tuple[float, dict[str, Any]]]:
        """`(seconds since the request was sent, chunk)` for every streamed chunk."""
        started = time.perf_counter()
        events = []
        with self._open(path, {**body, "stream": True}) as response:
            for line in response:
                if line.strip():
                    events.append((time.perf_counter() - started, json.loads(line)))
        return events


def chat_body(model: str, prompt: str, keep_alive: str, thinking: bool) -> dict[str, Any]:
    """A chat request; `think: false` only for models that declare the capability."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": OPTIONS,
        "keep_alive": keep_alive,
    }
    if thinking:
        body["think"] = False
    return body


def capabilities(client: Ollama, model: str) -> list[str]:
    return [str(c) for c in client.post("/api/show", {"model": model}).get("capabilities", [])]


def unload(client: Ollama, model: str, wait_s: float = 30) -> None:
    client.post("/api/generate", {"model": model, "keep_alive": 0})
    deadline = time.monotonic() + wait_s
    while model_size_vram_mb(client.get("/api/ps"), model) is not None:
        if time.monotonic() > deadline:
            raise OllamaError(f"{model} is still loaded {wait_s} s after unloading it")
        time.sleep(0.5)


def loaded_models(client: Ollama) -> set[str]:
    return {str(entry["name"]) for entry in client.get("/api/ps").get("models", [])}


def benchmark_model(
    client: Ollama, model: str, runs: int, sampler: VramSampler, ollama_version: str
) -> dict[str, Any]:
    software = model_software(client.get("/api/tags"), model)
    caps = capabilities(client, model)
    thinking = "thinking" in caps
    # Every loaded model, not only this one: the baseline must be an empty GPU.
    for loaded in sorted(loaded_models(client) | {model}):
        unload(client, loaded)
    baseline = sampler.used_mb()
    with sampler:
        body = chat_body(model, PROMPTS[0], WARM_KEEP_ALIVE, thinking)
        cold = parse_stream(client.stream("/api/chat", body))
        warm: list[StreamTimings] = []
        for index in range(runs):
            prompt = PROMPTS[1 + index % (len(PROMPTS) - 1)]
            body = chat_body(model, prompt, WARM_KEEP_ALIVE, thinking)
            warm.append(parse_stream(client.stream("/api/chat", body)))
    size_vram = model_size_vram_mb(client.get("/api/ps"), model)
    peak = sampler.peak_mb

    ttft = ColdWarm(cold.ttft_s, [w.ttft_s for w in warm])
    measurements = {
        "ttft_s": ttft.to_dict(),
        "tokens_per_s": ColdWarm(cold.tokens_per_s, [w.tokens_per_s for w in warm]).to_dict(),
        "prompt_eval_s": ColdWarm(cold.prompt_eval_s, [w.prompt_eval_s for w in warm]).to_dict(),
        "load_s": {"cold": cold.load_s},
        "runs": {"cold": cold.to_dict(), "warm": [w.to_dict() for w in warm]},
        "vram_mb": {
            "baseline": baseline,
            "peak": peak,
            "peak_above_baseline": None if peak is None or baseline is None else peak - baseline,
            "ollama_size_vram": size_vram,
        },
    }
    prompts = json.dumps([SYSTEM_PROMPT, *PROMPTS]).encode()
    return build_result(
        component="llm",
        variant=slug(model),
        machine=collect_machine(),
        software={
            "python": platform.python_version(),
            "ollama": ollama_version,
            **software,
            "capabilities": caps,
            "vram_sampler": sampler.source,
        },
        settings={
            "endpoint": client.endpoint,
            "api": "/api/chat",
            "options": OPTIONS,
            "think": False if thinking else None,
            "warm_runs": runs,
            "keep_alive_warm": WARM_KEEP_ALIVE,
            "prompts_sha256": hashlib.sha256(prompts).hexdigest(),
            "ttft_definition": "client-side, request sent to first non-empty token chunk",
            "budget_statistic": "warm p95",
        },
        measurements=measurements,
        checks=[evaluate_budget(BUDGETS["llm_warm_ttft"], ttft.summary().p95).to_dict()],
    )


def main(argv: Sequence[str] | None = None, sampler: VramSampler | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--endpoint", default=OLLAMA)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    client = Ollama(args.endpoint)
    try:
        version = str(client.get("/api/version")["version"])
        tags = client.get("/api/tags")
        for model in args.models:
            model_software(tags, model)
    except OllamaError as error:
        print(
            f"{error}\nStart Ollama (the Ollama app or `ollama serve`) and retry.", file=sys.stderr
        )
        return 2
    except LookupError as error:
        print(error, file=sys.stderr)
        return 2

    sampler = sampler or VramSampler()
    statuses: list[str] = []
    for model in args.models:
        print(f"llm: {model} (1 cold + {args.runs} warm runs)", flush=True)
        try:
            result = benchmark_model(client, model, args.runs, sampler, version)
        except (OllamaError, ValueError) as error:
            print(f"llm: {model} failed: {error}", file=sys.stderr)
            return 1
        path = write_result(result, args.output_dir)
        summary = result["measurements"]["ttft_s"]["summary"]
        speed = result["measurements"]["tokens_per_s"]["summary"]
        print(
            f"llm: {model} warm TTFT p50 {summary['p50']:.3f} s p95 {summary['p95']:.3f} s, "
            f"{speed['p50']:.1f} tokens/s -> {path.name}"
        )
        statuses += [check["status"] for check in result["checks"]]
    # Budgets only warn; a failed check (none today for the LLM) would exit 1.
    return 1 if "fail" in statuses else 0


if __name__ == "__main__":
    sys.exit(main())
