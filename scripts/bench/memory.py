"""Memory benchmark: speech-to-text and each LLM on the GPU at the same time.

This is the one hard rule of the benchmarks (roadmap: everything on the GPU fits in
8 GB). With every Ollama model unloaded, the GPU's used VRAM is read as the
baseline; faster-whisper is loaded in this process and transcribes one fixture;
then each LLM is loaded through the host's Ollama (`num_ctx` 4096) and generates
once. The VRAM used above the baseline, speech-to-text plus that model, must stay
at or under `PARLORA_BENCH_VRAM_LIMIT_MB` (default 7680): above it the check fails
and the exit code is 1. Other GPU processes are listed by executable name, and the
RAM of the Compose stack (`just up`) and of the host's Ollama process is recorded.

Usage: python -m scripts.bench.memory [MODEL ...]
"""

import argparse
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from scripts.bench.common.budgets import evaluate_vram, vram_limit_mb
from scripts.bench.common.ollama import model_size_vram_mb
from scripts.bench.common.probes import (
    parse_compute_apps,
    parse_docker_stats,
    parse_ps_rss_mb,
    parse_tasklist_mb,
)
from scripts.bench.common.results import RESULTS_DIR, write_result
from scripts.bench.common.schema import build_result
from scripts.bench.gpu import VramSampler
from scripts.bench.host import OLLAMA, collect_machine
from scripts.bench.llm import (
    DEFAULT_MODELS,
    Ollama,
    OllamaError,
    capabilities,
    chat_body,
    loaded_models,
    unload,
)
from scripts.bench.stt import load_fixtures, versions, whisper_transcriber

SETTLE_S = 1.0


def _run(cmd: list[str]) -> str:
    executable = shutil.which(cmd[0])
    if not executable:
        return ""
    result = subprocess.run(
        [executable, *cmd[1:]], capture_output=True, text=True, check=False, timeout=60
    )
    return result.stdout if result.returncode == 0 else ""


def gpu_processes() -> list[dict[str, Any]]:
    query = "--query-compute-apps=pid,process_name,used_memory"
    return parse_compute_apps(_run(["nvidia-smi", query, "--format=csv,noheader"]))


def stack_ram_mb() -> dict[str, float]:
    return parse_docker_stats(
        _run(["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"])
    )


def ollama_ram_mb() -> float:
    if sys.platform == "win32":
        return parse_tasklist_mb(
            _run(["tasklist", "/FI", "IMAGENAME eq ollama*", "/FO", "CSV", "/NH"])
        )
    else:
        return parse_ps_rss_mb(_run(["ps", "-C", "ollama", "-o", "rss="]))


def measure(
    *,
    load_stt: Callable[[], Callable[[Any], str]],
    stt_audio: Any,
    load_llm: Callable[[str], int | None],
    unload_all: Callable[[], None],
    models: Sequence[str],
    sampler: Any,
    limit_mb: int,
    settle: Callable[[], None] = lambda: time.sleep(SETTLE_S),
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Baseline, then STT, then each LLM with STT loaded; one VRAM check per model."""
    unload_all()
    settle()
    baseline = sampler.used_mb()
    transcribe = load_stt()
    transcribe(stt_audio)
    settle()
    with_stt = sampler.used_mb()
    per_model, checks = [], []
    for model in models:
        with sampler:
            size_vram = load_llm(model)
        settle()
        used = max(sampler.used_mb(), sampler.peak_mb or 0)
        check = evaluate_vram(model, used_mb=used, baseline_mb=baseline, limit_mb=limit_mb)
        checks.append(check.to_dict())
        per_model.append(
            {
                "model": model,
                "used_mb": used,
                "above_baseline_mb": used - baseline,
                "llm_mb": used - with_stt,
                "ollama_size_vram_mb": size_vram,
                "status": check.status,
            }
        )
        unload_all()
        settle()
    measurements = {
        "vram_mb": {
            "baseline": baseline,
            "stt_above_baseline": with_stt - baseline,
            "limit": limit_mb,
        },
        "models": per_model,
    }
    return measurements, checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--endpoint", default=OLLAMA)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    client = Ollama(args.endpoint)
    try:
        ollama_version = str(client.get("/api/version")["version"])
    except OllamaError as error:
        print(f"{error}\nStart Ollama and retry.", file=sys.stderr)
        return 2
    sampler = VramSampler()
    if sampler.source is None:
        print("memory: no NVIDIA GPU found (NVML and nvidia-smi are missing)", file=sys.stderr)
        return 2

    def unload_all() -> None:
        for model in sorted(loaded_models(client)):
            unload(client, model)

    def load_llm(model: str) -> int | None:
        thinking = "thinking" in capabilities(client, model)
        body = chat_body(model, "Hello! How are you today?", "5m", thinking)
        client.post("/api/chat", body)
        return model_size_vram_mb(client.get("/api/ps"), model)

    limit = vram_limit_mb()
    print(f"memory: STT + {', '.join(args.models)}, limit {limit} MiB", flush=True)
    measurements, checks = measure(
        load_stt=whisper_transcriber,
        stt_audio=load_fixtures()[0].audio,
        load_llm=load_llm,
        unload_all=unload_all,
        models=args.models,
        sampler=sampler,
        limit_mb=limit,
    )
    measurements["gpu_processes"] = gpu_processes()
    measurements["ram_mb"] = {"compose_stack": stack_ram_mb(), "ollama": ollama_ram_mb()}
    result = build_result(
        component="memory",
        variant="native",
        machine=collect_machine(),
        software={
            "python": platform.python_version(),
            "ollama": ollama_version,
            **versions(),
            "vram_sampler": sampler.source,
        },
        settings={
            "stt": "faster-whisper large-v3-turbo int8 cuda, one transcription",
            "llm_options": {"num_ctx": 4096},
            "limit_env": "PARLORA_BENCH_VRAM_LIMIT_MB",
            "rule": "used VRAM minus baseline, STT and one LLM loaded together",
        },
        measurements=measurements,
        checks=checks,
    )
    path = write_result(result, args.output_dir)
    for entry in measurements["models"]:
        print(
            f"memory: STT + {entry['model']}: {entry['above_baseline_mb']} MiB above baseline "
            f"-> {entry['status']}"
        )
    print(f"memory: -> {path.name}")
    return exit_code_of(checks)


def exit_code_of(checks: list[dict[str, Any]]) -> int:
    return 1 if any(check["status"] == "fail" for check in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
