"""Text-to-speech benchmark: the server's Kokoro adapter on the CPU or the GPU.

The adapter synthesises each text chunk in one piece, so the time to first audio of
a reply is the synthesis time of its first sentence. The benchmark synthesises a
fixed set of short tutor replies, some with out-of-vocabulary words, and records per
sentence the synthesis time, the real-time factor (synthesis time / audio duration)
and the OOV words. The OOV mode is recorded: `cli` where an `espeak-ng` executable is on PATH (the
bench image), `silent` elsewhere (native Windows), which is flagged as a warning
because OOV words are then dropped. `--provider` picks onnxruntime's CPU or CUDA
provider; the providers onnxruntime actually used are recorded, and a CUDA run that
fell back to the CPU is flagged. CUDA results get a `-cuda` variant suffix.

Usage: python -m scripts.bench.tts [--provider cpu|cuda] [--runs N]
"""

import argparse
import platform
import sys
import time
from collections.abc import Callable, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from scripts.bench.common.budgets import BUDGETS, evaluate_budget
from scripts.bench.common.results import RESULTS_DIR, write_result
from scripts.bench.common.schema import build_result
from scripts.bench.common.stats import ColdWarm, real_time_factor
from scripts.bench.host import collect_machine
from scripts.bench.stt import detect_variant

DEFAULT_RUNS = 5
# Short spoken tutor replies; the last three hold words misaki does not know.
SENTENCES = (
    "That sounds great!",
    "Nice try, but we say I went, not I go.",
    "Can you tell me more about your weekend?",
    "Good question: say is about the words, tell is about the listener.",
    "Let's practise ordering food in a restaurant.",
    "You are doing really well today.",
    "Try again, and this time use the past tense.",
    "Did you enjoy your trip to Zürich?",
    "Siobhan and Niamh are Irish names.",
    "The word Parlora is spelled with one r in the middle.",
)


PROVIDERS = ("cpu", "cuda")
ONNX_PROVIDER = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}


class Synthesiser(Protocol):
    oov_mode: str
    providers: tuple[str, ...]

    def __call__(self, text: str) -> tuple[float, tuple[str, ...]]:
        """Synthesise `text`; return the audio duration in seconds and the OOV words."""
        ...


def kokoro_synthesiser(provider: str) -> Synthesiser:
    from scripts.bench.kokoro_runner import Kokoro

    kokoro = Kokoro(provider="cuda" if provider == "cuda" else "cpu")

    class Run:
        oov_mode = kokoro.oov_mode
        providers = kokoro.providers

        def __call__(self, text: str) -> tuple[float, tuple[str, ...]]:
            speech = kokoro.synthesize(text)
            return speech.duration_s, speech.oov_words

    return Run()


def measure(
    load: Callable[[], Synthesiser],
    sentences: Sequence[str],
    runs: int,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    started = clock()
    synthesise = load()
    load_s = clock() - started
    started = clock()
    synthesise(sentences[0])
    cold = clock() - started

    times: list[float] = []
    rtfs: list[float] = []
    per_sentence = []
    oov: list[str] = []
    for sentence in sentences:
        sentence_times: list[float] = []
        duration: float = 0.0
        words: tuple[str, ...] = ()
        for _ in range(runs):
            started = clock()
            duration, words = synthesise(sentence)
            sentence_times.append(clock() - started)
        rtf = [real_time_factor(processing_s=t, audio_s=duration) for t in sentence_times]
        times += sentence_times
        rtfs += rtf
        oov += words
        per_sentence.append(
            {
                "text": sentence,
                "duration_s": round(duration, 3),
                "synthesis_s": ColdWarm(None, sentence_times).to_dict(),
                "oov_words": list(words),
            }
        )
    return {
        "load_s": load_s,
        "first_audio_s": ColdWarm(cold, times).to_dict(),
        "rtf": ColdWarm(None, rtfs).to_dict(),
        "oov": {"mode": synthesise.oov_mode, "count": len(oov), "words": sorted(set(oov))},
        "onnx_providers": list(synthesise.providers),
        "sentences": per_sentence,
    }


def checks_for(measurements: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    first = measurements["first_audio_s"]["summary"]["p95"]
    rtf = measurements["rtf"]["summary"]["p95"]
    checks = [
        evaluate_budget(BUDGETS["tts_first_audio"], first).to_dict(),
        evaluate_budget(BUDGETS["tts_rtf"], rtf).to_dict(),
    ]
    mode = measurements["oov"]["mode"]
    checks.append(
        {
            "name": "tts_oov_mode",
            "status": "pass" if mode == "cli" else "warn",
            "value": mode,
            "detail": "silent: no espeak-ng on PATH, OOV words are dropped (ADR 0004)",
        }
    )
    used = measurements["onnx_providers"][0] if measurements["onnx_providers"] else None
    checks.append(
        {
            "name": "tts_provider",
            "status": "pass" if used == ONNX_PROVIDER[provider] else "warn",
            "value": used,
            "detail": f"{provider} requested; onnxruntime ran on the first provider listed",
        }
    )
    return checks


def versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in ("onnxruntime", "misaki", "spacy", "numpy"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def main(
    argv: Sequence[str] | None = None, load: Callable[[str], Synthesiser] = kokoro_synthesiser
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider", choices=PROVIDERS, default="cpu")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--variant", choices=["native", "container"], default=detect_variant())
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    variant = args.variant if args.provider == "cpu" else f"{args.variant}-{args.provider}"
    print(f"tts: {variant}, {len(SENTENCES)} sentences x {args.runs} runs", flush=True)
    measurements = measure(lambda: load(args.provider), SENTENCES, args.runs)
    result = build_result(
        component="tts",
        variant=variant,
        machine=collect_machine(),
        software={"python": platform.python_version(), **versions()},
        settings={
            "model": "kokoro-v1.0.onnx",
            "provider": args.provider,
            "onnx_providers": measurements["onnx_providers"],
            "voice": "af_heart",
            "speed": 1.0,
            "runs_per_sentence": args.runs,
            "oov_mode": measurements["oov"]["mode"],
            "first_audio_definition": "synthesis time of one sentence (one chunk per call)",
            "budget_statistic": "p95",
        },
        measurements=measurements,
        checks=checks_for(measurements, args.provider),
    )
    path = write_result(result, args.output_dir)
    first = measurements["first_audio_s"]["summary"]
    print(
        f"tts: first audio p50 {first['p50']:.3f} s p95 {first['p95']:.3f} s, "
        f"RTF p50 {measurements['rtf']['summary']['p50']:.2f}, OOV mode "
        f"{measurements['oov']['mode']} ({measurements['oov']['count']} words) -> {path.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
