"""Parse Ollama's native API responses into benchmark timings.

Time to first token (TTFT) is measured on the client: from sending the request to
receiving the first chunk whose text is not empty. Ollama's own durations come from
the final streamed chunk, in nanoseconds: `load_duration`, `prompt_eval_duration`
with `prompt_eval_count`, and `eval_duration` with `eval_count`. Generation speed is
`eval_count / eval_duration`.

Events are `(seconds since the request was sent, decoded JSON chunk)`.
"""

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

NS = 1_000_000_000
MB = 1024 * 1024


@dataclass(frozen=True)
class StreamTimings:
    ttft_s: float
    total_s: float
    load_s: float
    prompt_eval_s: float
    prompt_eval_count: int
    eval_s: float
    eval_count: int
    tokens_per_s: float
    done_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(chunk: Mapping[str, Any]) -> str:
    """Token text of a /api/chat (`message.content`) or /api/generate (`response`) chunk."""
    message = chunk.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(chunk.get("response") or "")


def parse_stream(events: Iterable[tuple[float, Mapping[str, Any]]]) -> StreamTimings:
    ttft: float | None = None
    final: Mapping[str, Any] | None = None
    for elapsed, chunk in events:
        if "error" in chunk:
            raise ValueError(f"Ollama returned an error: {chunk['error']}")
        if ttft is None and _text(chunk):
            ttft = elapsed
        if chunk.get("done"):
            final = chunk
    if ttft is None:
        raise ValueError("the stream holds no token")
    if final is None:
        raise ValueError("the stream ended without a final chunk")
    eval_ns = int(final.get("eval_duration", 0))
    if eval_ns <= 0:
        raise ValueError("the final chunk has no eval_duration")
    eval_count = int(final.get("eval_count", 0))
    return StreamTimings(
        ttft_s=ttft,
        total_s=int(final.get("total_duration", 0)) / NS,
        load_s=int(final.get("load_duration", 0)) / NS,
        prompt_eval_s=int(final.get("prompt_eval_duration", 0)) / NS,
        prompt_eval_count=int(final.get("prompt_eval_count", 0)),
        eval_s=eval_ns / NS,
        eval_count=eval_count,
        tokens_per_s=eval_count / (eval_ns / NS),
        done_reason=final.get("done_reason"),
    )


def model_size_vram_mb(ps: Mapping[str, Any], model: str) -> int | None:
    """VRAM Ollama reports for a loaded model in `/api/ps`, in MiB; `None` if not loaded."""
    for entry in ps.get("models", []):
        if model in (entry.get("name"), entry.get("model")):
            return int(entry.get("size_vram", 0)) // MB
    return None


def model_software(tags: Mapping[str, Any], model: str) -> dict[str, Any]:
    """Tag, digest, size and quantisation of an installed model, from `/api/tags`."""
    for entry in tags.get("models", []):
        if model in (entry.get("name"), entry.get("model")):
            details = entry.get("details", {})
            return {
                "model": model,
                "digest": entry.get("digest"),
                "size_mb": int(entry.get("size", 0)) // MB,
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
            }
    raise LookupError(f"model {model} is not installed; run `ollama pull {model}`")
