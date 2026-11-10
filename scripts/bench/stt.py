"""Speech-to-text benchmark: faster-whisper on the GPU, natively or in the container.

The same script runs on the host (`just bench-stt`) and in the bench image
(`just bench-stt-container`), so the two variants cannot drift apart (ADR 0002's
open question). Settings follow ADR 0002: `large-v3-turbo`, `int8` on CUDA, beam
size 5, audio fed as in-memory float32 PCM (the committed synthetic fixtures).

Measured: model load, the first (warm-up) transcription, N warm transcriptions per
fixture (p50/p95 latency, real-time factor), word error rate against the
reference text after normalisation, and the GPU's used VRAM after loading and at
peak, minus the baseline before loading. If CUDA cannot be used, the error is
recorded in the result instead of crashing, and the exit code is 1.

Usage: python -m scripts.bench.stt [--runs N] [--variant native|container]
"""

import argparse
import json
import os
import platform
import sys
import time
import wave
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from parlora_server.adapters.cuda import add_cuda_dll_directories
from scripts.bench.common.budgets import BUDGETS, evaluate_budget
from scripts.bench.common.results import RESULTS_DIR, write_result
from scripts.bench.common.schema import build_result
from scripts.bench.common.stats import ColdWarm, real_time_factor, summarise
from scripts.bench.common.wer import score
from scripts.bench.fixtures import FIXTURES
from scripts.bench.gpu import VramSampler
from scripts.bench.host import collect_machine

MODEL = "large-v3-turbo"
SETTINGS = {"model": MODEL, "device": "cuda", "compute_type": "int8", "beam_size": 5}
DEFAULT_RUNS = 10
VARIANT_ENV = "PARLORA_BENCH_VARIANT"
NVIDIA_WHEELS = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12")


class Transcriber(Protocol):
    def __call__(self, audio: Any) -> str: ...


@dataclass(frozen=True)
class Fixture:
    id: str
    cefr: str
    text: str
    audio: Any  # float32 NumPy array at 16 kHz
    duration_s: float


def load_fixtures(directory: Path = FIXTURES) -> list[Fixture]:
    import numpy as np

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    fixtures = []
    for entry in manifest["utterances"]:
        with wave.open(str(directory / f"{entry['id']}.wav"), "rb") as file:
            rate, frames = file.getframerate(), file.readframes(file.getnframes())
        audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        fixture = Fixture(entry["id"], entry["cefr"], entry["text"], audio, len(audio) / rate)
        fixtures.append(fixture)
    return fixtures


def whisper_transcriber() -> Transcriber:
    add_cuda_dll_directories()
    from faster_whisper import WhisperModel

    model = WhisperModel(MODEL, device=SETTINGS["device"], compute_type=SETTINGS["compute_type"])

    def transcribe(audio: Any) -> str:
        segments, _ = model.transcribe(audio, beam_size=SETTINGS["beam_size"], language="en")
        return "".join(segment.text for segment in segments).strip()

    return transcribe


def detect_variant() -> str:
    if os.environ.get(VARIANT_ENV):
        return os.environ[VARIANT_ENV]
    return "container" if Path("/.dockerenv").exists() else "native"


def measure(
    load: Callable[[], Transcriber],
    fixtures: Sequence[Fixture],
    runs: int,
    sampler: Any,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Load, warm up on the first fixture, then `runs` timed transcriptions per fixture."""
    baseline = sampler.used_mb()
    with sampler:
        started = clock()
        transcribe = load()
        load_s = clock() - started
        after_load = sampler.used_mb()
        started = clock()
        transcribe(fixtures[0].audio)
        first_s = clock() - started

        per_fixture = []
        latencies: list[float] = []
        rtfs: list[float] = []
        errors = words = 0
        for fixture in fixtures:
            times, hypothesis = [], ""
            for _ in range(runs):
                started = clock()
                hypothesis = transcribe(fixture.audio)
                times.append(clock() - started)
            word_errors = score(fixture.text, hypothesis)
            errors += word_errors.substitutions + word_errors.deletions + word_errors.insertions
            words += word_errors.reference_words
            p50 = summarise(times).p50
            rtfs.append(real_time_factor(processing_s=p50, audio_s=fixture.duration_s))
            per_fixture.append(
                {
                    "id": fixture.id,
                    "cefr": fixture.cefr,
                    "duration_s": round(fixture.duration_s, 3),
                    "latency_s": ColdWarm(None, times).to_dict(),
                    "rtf_p50": rtfs[-1],
                    "wer": word_errors.rate,
                    "hypothesis": hypothesis,
                }
            )
            latencies += times
    peak = sampler.peak_mb
    return {
        "load_s": load_s,
        "first_transcription_s": first_s,
        "latency_s": ColdWarm(first_s, latencies).to_dict(),
        "rtf": summarise(rtfs).to_dict(),
        "wer": {"overall": errors / words, "label": "synthetic speech"},
        "fixtures": per_fixture,
        "vram_mb": {
            "baseline": baseline,
            "after_load": None if after_load is None else after_load - (baseline or 0),
            "peak": None if peak is None else peak - (baseline or 0),
        },
    }


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in ("faster-whisper", "ctranslate2", *NVIDIA_WHEELS):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def main(
    argv: Sequence[str] | None = None,
    load: Callable[[], Transcriber] = whisper_transcriber,
    sampler: Any = None,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--variant", choices=["native", "container"], default=detect_variant())
    parser.add_argument("--fixtures", type=Path, default=FIXTURES)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    fixtures = load_fixtures(args.fixtures)
    sampler = sampler or VramSampler()
    software = {"python": platform.python_version(), **versions(), "vram_sampler": sampler.source}
    settings = {**SETTINGS, "warm_runs_per_fixture": args.runs, "input": "float32 PCM"}
    print(f"stt: {args.variant}, {len(fixtures)} fixtures x {args.runs} warm runs", flush=True)
    try:
        measurements = measure(load, fixtures, args.runs, sampler)
        p95 = measurements["latency_s"]["summary"]["p95"]
        checks = [evaluate_budget(BUDGETS["stt_warm_p95"], p95).to_dict()]
        code = 0
    except Exception as error:  # CUDA or driver failure: record it, do not crash
        measurements = {"error": f"{type(error).__name__}: {error}"}
        checks = [{"name": "stt_cuda", "status": "fail", "detail": measurements["error"]}]
        code = 1
    result = build_result(
        component="stt",
        variant=args.variant,
        machine=collect_machine(cuda_runtime=software.get("nvidia-cublas-cu12")),
        software=software,
        settings=settings,
        measurements=measurements,
        checks=checks,
    )
    path = write_result(result, args.output_dir)
    if code:
        print(
            f"stt: failed, error recorded in {path.name}: {measurements['error']}", file=sys.stderr
        )
    else:
        latency = measurements["latency_s"]["summary"]
        print(
            f"stt: warm p50 {latency['p50']:.3f} s p95 {latency['p95']:.3f} s, "
            f"WER {measurements['wer']['overall']:.3f}, load {measurements['load_s']:.1f} s "
            f"-> {path.name}"
        )
    return code


if __name__ == "__main__":
    sys.exit(main())
