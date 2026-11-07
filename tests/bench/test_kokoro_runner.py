"""The Kokoro runner's pure parts: weights, OOV routing, IPA mapping, trimming."""

import hashlib
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.bench import kokoro_runner as kr

pytestmark = pytest.mark.unit


# ── cache and weights ───────────────────────────────────────────────────────


def test_cache_dir_from_the_environment(tmp_path: Path) -> None:
    assert kr.cache_dir({kr.CACHE_ENV: str(tmp_path)}) == tmp_path / "kokoro"


def test_cache_dir_defaults_per_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    assert kr.cache_dir({"LOCALAPPDATA": str(tmp_path)}) == tmp_path / "parlora/bench/kokoro"
    monkeypatch.setattr(sys, "platform", "linux")
    assert kr.cache_dir({"XDG_CACHE_HOME": str(tmp_path)}) == tmp_path / "parlora/bench/kokoro"
    assert kr.cache_dir({}) == Path.home() / ".cache/parlora/bench/kokoro"


def test_weights_are_pinned_https_downloads() -> None:
    assert set(kr.WEIGHTS) == {"kokoro-v1.0.onnx", "voices-v1.0.bin", "config.json"}
    for weight in kr.WEIGHTS.values():
        assert weight.url.startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", weight.sha256)
    assert "/resolve/main/" not in kr.WEIGHTS["config.json"].url


def weight(content: bytes) -> kr.Weight:
    return kr.Weight(
        "model.onnx", "https://example.test/model.onnx", hashlib.sha256(content).hexdigest()
    )


def downloader(content: bytes, calls: list[str]) -> Any:
    def download(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(content)

    return download


def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    calls: list[str] = []

    path = kr.fetch(weight(b"weights"), tmp_path / "cache", downloader(b"weights", calls))

    assert path.read_bytes() == b"weights"
    assert calls == ["https://example.test/model.onnx"]
    assert not list((tmp_path / "cache").glob("*.part"))


def test_fetch_reuses_a_verified_copy(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"weights")
    calls: list[str] = []

    kr.fetch(weight(b"weights"), tmp_path, downloader(b"weights", calls))

    assert calls == []


def test_fetch_replaces_a_corrupt_copy(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"truncated")
    calls: list[str] = []

    kr.fetch(weight(b"weights"), tmp_path, downloader(b"weights", calls))

    assert (tmp_path / "model.onnx").read_bytes() == b"weights"
    assert len(calls) == 1


def test_fetch_rejects_a_wrong_hash(tmp_path: Path) -> None:
    with pytest.raises(kr.WeightError, match="SHA-256"):
        kr.fetch(weight(b"weights"), tmp_path, downloader(b"tampered", []))
    assert list(tmp_path.iterdir()) == []


def test_downloads_need_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="https"):
        kr._download("http://example.test/model.onnx", tmp_path / "x")


# ── OOV words through espeak-ng ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("ipa", "kokoro"),
    [
        ("zˈuːɹɪt^ʃ", "zˈuɹɪʧ"),  # Zürich, espeak-ng 1.52 en-us
        ("nˈa^ɪntiːn", "nˈIntin"),  # nineteen
        ("pˈɑː^ɹlɔːɹə", "pˈɑɹlɔɹə"),  # Parlora
        (" d^ʒˈe^ɪn\n", "ʤˈAn"),  # Jane, with the whitespace espeak-ng prints
        ("bˈɜːd", "bˈɜɹd"),
        ("ɡˈo^ʊ", "ɡˈO"),
    ],
)
def test_ipa_to_kokoro(ipa: str, kokoro: str) -> None:
    assert kr.ipa_to_kokoro(ipa) == kokoro


def test_ipa_to_kokoro_british() -> None:
    assert kr.ipa_to_kokoro("ɡˈə^ʊ", british=True) == "ɡˈQ"


def completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


def test_espeak_cli_phonemises_one_word_per_process() -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        assert kwargs["timeout"] > 0
        assert kwargs["check"] is False
        return completed(" zˈuːɹɪt^ʃ\n")

    fallback = kr.EspeakCli("/usr/bin/espeak-ng", run=run)

    assert fallback(SimpleNamespace(text="Zürich")) == ("zˈuɹɪʧ", 2)
    assert commands == [["/usr/bin/espeak-ng", "-q", "--ipa", "--tie=^", "-v", "en-us", "Zürich"]]
    assert fallback.words == ["Zürich"]


@pytest.mark.parametrize("result", [completed("", 0), completed("zˈuːɹɪk", 1)])
def test_espeak_cli_failure_leaves_the_word_silent(
    result: subprocess.CompletedProcess[str],
) -> None:
    fallback = kr.EspeakCli("espeak-ng", run=lambda *a, **k: result)

    assert fallback(SimpleNamespace(text="Qwxz")) == (None, None)
    assert fallback.words == ["Qwxz"]


def test_cli_mode_only_when_espeak_ng_is_on_path() -> None:
    assert kr.espeak_fallback(lambda name: None) is None
    fallback = kr.espeak_fallback(lambda name: f"/usr/bin/{name}")
    assert isinstance(fallback, kr.EspeakCli)
    assert fallback.executable == "/usr/bin/espeak-ng"


# ── trimming ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("levels", "bounds"),
    [
        ([0, 0, 5, 6, 0, 7, 0, 0], (2, 6)),
        ([9, 9], (0, 2)),
        ([0, 0, 0], (0, 0)),
        ([], (0, 0)),
    ],
)
def test_speech_bounds(levels: list[float], bounds: tuple[int, int]) -> None:
    assert kr.speech_bounds(levels, threshold=1.0) == bounds


# ── real synthesis (bench group and cached weights) ─────────────────────────


@pytest.mark.slow
def test_real_synthesis_produces_trimmed_speech() -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("misaki")
    if not all((kr.cache_dir() / name).is_file() for name in kr.WEIGHTS):
        pytest.skip("Kokoro weights are not cached; run `just bench-fixtures` once")

    speech = kr.Kokoro().synthesize("Hello, my name is Anna and I live in Zürich.")

    assert 1.5 < speech.duration_s < 6.0
    assert speech.phonemes
