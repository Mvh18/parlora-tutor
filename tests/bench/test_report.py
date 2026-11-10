"""The benchmark report renderer, against fixture results and a golden file."""

import json
from pathlib import Path

import pytest

from scripts.bench import report

pytestmark = pytest.mark.unit

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures" / "results"
GOLDEN = HERE / "golden" / "benchmarks.md.golden"


def test_report_from_fixture_results_matches_the_golden_file() -> None:
    assert report.render(report.load(FIXTURES)) == GOLDEN.read_text(encoding="utf-8")


def test_report_flags_silent_mode_errors_and_labels_wer() -> None:
    text = report.render(report.load(FIXTURES))

    assert "silent ⚠️ OOV words dropped" in text
    assert "❌ RuntimeError: CUDA driver version is insufficient" in text
    assert "WER (synthetic speech)" in text
    assert "Measured on the client, warm runs only" in text


def test_only_the_latest_result_per_component_and_variant_is_shown() -> None:
    old = {"component": "llm", "variant": "m", "created_at": "2026-01-01T00:00:00Z"}
    new = {**old, "created_at": "2026-02-01T00:00:00Z"}
    other = {**old, "variant": "n"}

    chosen = report.latest([("new.json", new), ("old.json", old), ("other.json", other)])

    assert chosen == {("llm", "m"): ("new.json", new), ("llm", "n"): ("other.json", other)}


def test_empty_results_still_render() -> None:
    text = report.render([])

    assert text.startswith("# Component benchmarks\n")
    assert "## LLM" not in text


def test_check_mode_detects_a_stale_report(tmp_path: Path) -> None:
    output = tmp_path / "benchmarks.md"
    args = ["--results", str(FIXTURES), "--output", str(output)]

    assert report.main([*args, "--check"]) == 1
    assert report.main(args) == 0
    assert report.main([*args, "--check"]) == 0
    output.write_text("edited by hand\n", encoding="utf-8")
    assert report.main([*args, "--check"]) == 1


def test_links_point_at_the_result_files() -> None:
    text = report.render(report.load(FIXTURES))

    for path in FIXTURES.glob("*.json"):
        assert f"(../benchmarks/results/{path.name})" in text
    assert json.loads((FIXTURES / next(FIXTURES.glob("*-llm-*.json")).name).read_text("utf-8"))


def test_memory_section_shows_kokoro_when_measured() -> None:
    results = report.load(FIXTURES)
    (name, memory), *_ = [(n, r) for n, r in results if r["component"] == "memory"]
    with_tts = json.loads(json.dumps(memory))
    with_tts["measurements"]["vram_mb"]["tts_above_stt"] = 600
    others = [(n, r) for n, r in results if r["component"] != "memory"]

    assert "Kokoro adds" not in report.render(results)
    assert "Kokoro adds 600 MiB" in report.render([*others, (name, with_tts)])
