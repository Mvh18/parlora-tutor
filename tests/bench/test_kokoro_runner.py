"""The benchmark's Kokoro runner: the server adapter plus the benchmark weight cache.

The adapter's own logic (weights, OOV routing, IPA mapping, trimming, providers) is
tested with the server, in `apps/server/tests/unit/test_kokoro_tts.py`.
"""

import sys
from pathlib import Path

import pytest

from parlora_server.adapters.tts import kokoro as adapter
from scripts.bench import kokoro_runner as kr

pytestmark = pytest.mark.unit


def test_cache_dir_from_the_environment(tmp_path: Path) -> None:
    assert kr.cache_dir({kr.CACHE_ENV: str(tmp_path)}) == tmp_path / "kokoro"


def test_cache_dir_defaults_per_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert kr.cache_dir({"LOCALAPPDATA": str(tmp_path)}) == tmp_path / "parlora/bench/kokoro"
    monkeypatch.setattr(sys, "platform", "linux")
    assert kr.cache_dir({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "parlora/bench/kokoro"
    assert kr.cache_dir({}) == Path.home() / ".cache/parlora/bench/kokoro"


def test_the_runner_is_the_server_adapter() -> None:
    assert issubclass(kr.Kokoro, adapter.Kokoro)
    for name in ("WEIGHTS", "SAMPLE_RATE", "EspeakCli", "fetch", "ipa_to_kokoro", "speech_bounds"):
        assert getattr(kr, name) is getattr(adapter, name)


def test_the_runner_defaults_to_the_bench_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: list[Path] = []

    def init(self: adapter.Kokoro, directory: Path, **kwargs: object) -> None:
        seen.append(directory)

    monkeypatch.setattr(adapter.Kokoro, "__init__", init)
    monkeypatch.setenv(kr.CACHE_ENV, str(tmp_path))

    kr.Kokoro()
    kr.Kokoro(tmp_path / "elsewhere")

    assert seen == [tmp_path / "kokoro", tmp_path / "elsewhere"]


# ── real synthesis (speech dependencies and cached weights) ─────────────────


@pytest.mark.slow
def test_real_synthesis_produces_trimmed_speech() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("misaki")
    if not all((kr.cache_dir() / name).is_file() for name in kr.WEIGHTS):
        pytest.skip("Kokoro weights are not cached; run `just bench-fixtures` once")

    speech = kr.Kokoro().synthesize("Hello, my name is Anna and I live in Zürich.")

    assert 1.5 < speech.duration_s < 6.0
    assert speech.phonemes
