"""Render docs/benchmarks.md from the latest result of each component and variant.

The report is generated, never edited by hand: `just bench-report` rewrites it and
a repository test fails when the committed file differs from what the results in
benchmarks/results/ produce. Standard library only.

Usage: python -m scripts.bench.report [--check]
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.bench.common.results import RESULTS_DIR

REPORT = Path(__file__).resolve().parents[2] / "docs" / "benchmarks.md"
ICONS = {"pass": "✅", "warn": "⚠️", "fail": "❌"}
COMPONENTS = ("llm", "stt", "tts", "memory")

DEFINITIONS = """\
- **Cold / warm.** Each benchmark reports one cold run on its own and summarises
  the warm runs that follow. Percentiles use linear interpolation between the
  closest ranks (NumPy's default).
- **Time to first token (TTFT).** Measured on the client, warm runs only: from
  sending the request to Ollama's native API to the first chunk with text.
- **Time to first audio.** The Kokoro runner does not stream, so it is the
  synthesis time of one sentence of a reply.
- **Real-time factor (RTF).** Processing time divided by audio duration; below
  1.0 is faster than real time.
- **Word error rate (WER).** On synthetic speech (the committed Kokoro fixtures)
  after normalising case, punctuation, contractions, numbers and accents. It is
  optimistic and not used for decisions.
- **VRAM above baseline.** The GPU's used memory minus what was in use before
  the models were loaded (desktop and other processes).
- **Budgets** only warn; the VRAM limit (`PARLORA_BENCH_VRAM_LIMIT_MB`, default
  7680 MiB) fails the memory benchmark."""


def latest(
    results: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[tuple[str, str], tuple[str, Mapping[str, Any]]]:
    """The newest result per (component, variant), by `created_at` then file name."""
    chosen: dict[tuple[str, str], tuple[str, Mapping[str, Any]]] = {}
    for name, result in sorted(results, key=lambda item: (item[1]["created_at"], item[0])):
        chosen[(result["component"], result["variant"])] = (name, result)
    return chosen


def load(directory: Path) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (path.name, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


def _s(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f} s"


def _mb(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f} MiB"


def _status(result: Mapping[str, Any], name: str) -> str:
    for check in result["checks"]:
        if check["name"] == name:
            return ICONS[check["status"]]
    return "—"


def _link(name: str) -> str:
    return f"[{name.removesuffix('.json')}](../benchmarks/results/{name})"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    return lines + ["| " + " | ".join(row) + " |" for row in rows]


def _llm(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    rows = []
    for name, r in entries:
        m = r["measurements"]
        ttft, speed = m["ttft_s"]["summary"], m["tokens_per_s"]["summary"]
        rows.append(
            [
                f"`{r['software']['model']}`",
                f"{_s(ttft['p50'])} / {_s(ttft['p95'])}",
                f"{speed['p50']:.1f}",
                _mb(m["vram_mb"]["peak_above_baseline"]),
                _s(m["load_s"]["cold"], 1),
                _status(r, "llm_warm_ttft"),
                _link(name),
            ]
        )
    header = ["Model", "Warm TTFT p50 / p95", "Tokens/s p50", "VRAM above baseline", "Cold load"]
    return _table([*header, "TTFT ≤ 0.4 s", "Result"], rows)


def _stt(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    rows = []
    for name, r in entries:
        m = r["measurements"]
        if "error" in m:
            rows.append([r["variant"], f"❌ {m['error']}", "", "", "", "", "", _link(name)])
            continue
        warm = m["latency_s"]["summary"]
        rows.append(
            [
                r["variant"],
                _s(m["load_s"], 1),
                f"{_s(warm['p50'])} / {_s(warm['p95'])}",
                f"{m['rtf']['p50']:.3f}",
                f"{m['wer']['overall']:.3f}",
                _mb(m["vram_mb"]["peak"]),
                _status(r, "stt_warm_p95"),
                _link(name),
            ]
        )
    header = ["Variant", "Load", "Warm p50 / p95", "RTF p50", "WER (synthetic speech)"]
    return _table([*header, "VRAM peak above baseline", "p95 ≤ 1.0 s", "Result"], rows)


def _tts(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    rows = []
    for name, r in entries:
        m = r["measurements"]
        first, mode = m["first_audio_s"]["summary"], m["oov"]["mode"]
        rows.append(
            [
                r["variant"],
                f"{mode} ⚠️ OOV words dropped" if mode == "silent" else mode,
                f"{_s(first['p50'])} / {_s(first['p95'])}",
                f"{m['rtf']['summary']['p50']:.2f}",
                str(m["oov"]["count"]),
                _status(r, "tts_first_audio"),
                _status(r, "tts_rtf"),
                _link(name),
            ]
        )
    header = ["Variant", "OOV mode", "First audio p50 / p95", "RTF p50", "OOV words"]
    return _table([*header, "First audio ≤ 0.5 s", "RTF ≤ 1.0", "Result"], rows)


def _memory(entries: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for name, r in entries:
        m = r["measurements"]
        vram = m["vram_mb"]
        lines += [
            f"Baseline {_mb(vram['baseline'])}, speech-to-text adds "
            f"{_mb(vram['stt_above_baseline'])}, limit {_mb(vram['limit'])} ({_link(name)}).",
            "",
        ]
        rows = [
            [f"`{x['model']}`", _mb(x["above_baseline_mb"]), _mb(x["used_mb"]), ICONS[x["status"]]]
            for x in m["models"]
        ]
        lines += _table(["Model", "STT + LLM above baseline", "Total used", "Within limit"], rows)
        ram = m.get("ram_mb", {})
        stack = ram.get("compose_stack", {})
        if stack or ram.get("ollama") is not None:
            parts = [f"{container} {_mb(value)}" for container, value in sorted(stack.items())]
            lines += [
                "",
                f"RAM: {', '.join(parts) or 'stack not running'}; "
                f"host Ollama {_mb(ram.get('ollama'))}.",
            ]
    return lines


SECTIONS = {
    "llm": ("LLM (host Ollama)", _llm),
    "stt": ("Speech-to-text (faster-whisper large-v3-turbo, int8, CUDA)", _stt),
    "tts": ("Text-to-speech (Kokoro ONNX on the CPU)", _tts),
    "memory": ("Memory: speech-to-text and each LLM on the GPU together", _memory),
}


def render(results: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    chosen = latest(results)
    lines = [
        "# Component benchmarks",
        "",
        "<!-- Generated by `just bench-report` from benchmarks/results/. Do not edit. -->",
        "",
        "Measured on the target machine with the scripts in `scripts/bench/`. "
        "Decisions taken from these numbers are in "
        "[ADR 0009](adr/0009-phase-1-llm-and-vram-budget.md).",
        "",
    ]
    if chosen:
        machine = max(chosen.values(), key=lambda item: item[1]["created_at"])[1]["machine"]
        facts = [machine.get(k) for k in ("gpu", "driver", "cpu", "os", "os_version")]
        ram = machine.get("ram_total_mb")
        vram = machine.get("vram_total_mb")
        lines += [
            f"**Machine:** {facts[0]} ({_mb(vram)}, driver {facts[1]}), {facts[2]}, "
            f"{_mb(ram)} RAM, {facts[3]} {facts[4]}.",
            "",
        ]
    lines += ["## Definitions", "", DEFINITIONS, ""]
    for component in COMPONENTS:
        entries = [chosen[key] for key in sorted(chosen) if key[0] == component]
        if not entries:
            continue
        title, section = SECTIONS[component]
        lines += [f"## {title}", "", *section(entries), ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the report is stale")
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    text = render(load(args.results))
    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.is_file() else ""
        if current != text:
            print(f"{args.output} is stale; run `just bench-report`", file=sys.stderr)
            return 1
        return 0
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(f"report: wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
