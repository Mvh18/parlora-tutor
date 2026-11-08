"""The Kokoro text-to-speech adapter with a fake onnxruntime, G2P and weights."""

import asyncio
import hashlib
import json
import logging
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from parlora_core.audio import AudioChunk
from parlora_core.errors import UnsupportedLanguage
from parlora_core.languages import LanguageCode
from parlora_core.ports import TextToSpeech
from parlora_server.adapters.tts import kokoro as kk

pytestmark = pytest.mark.unit

EN = LanguageCode("en")


# ── weights ─────────────────────────────────────────────────────────────────


def test_weights_are_pinned_https_downloads() -> None:
    assert set(kk.WEIGHTS) == {"kokoro-v1.0.onnx", "voices-v1.0.bin", "config.json"}
    for weight in kk.WEIGHTS.values():
        assert weight.url.startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{64}", weight.sha256)
    assert "/resolve/main/" not in kk.WEIGHTS["config.json"].url


def weight(content: bytes, name: str = "model.onnx") -> kk.Weight:
    return kk.Weight(name, f"https://example.test/{name}", hashlib.sha256(content).hexdigest())


def downloader(content: bytes, calls: list[str]) -> Callable[[str, Path], None]:
    def download(url: str, target: Path) -> None:
        calls.append(url)
        target.write_bytes(content)

    return download


def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    calls: list[str] = []

    path = kk.fetch(weight(b"weights"), tmp_path / "cache", downloader(b"weights", calls))

    assert path.read_bytes() == b"weights"
    assert calls == ["https://example.test/model.onnx"]
    assert not list((tmp_path / "cache").glob("*.part"))


def test_fetch_reuses_a_verified_copy(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"weights")
    calls: list[str] = []

    kk.fetch(weight(b"weights"), tmp_path, downloader(b"weights", calls))

    assert calls == []


def test_fetch_replaces_a_corrupt_copy(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"truncated")
    calls: list[str] = []

    kk.fetch(weight(b"weights"), tmp_path, downloader(b"weights", calls))

    assert (tmp_path / "model.onnx").read_bytes() == b"weights"
    assert len(calls) == 1


def test_fetch_rejects_a_wrong_hash(tmp_path: Path) -> None:
    with pytest.raises(kk.WeightError, match="SHA-256"):
        kk.fetch(weight(b"weights"), tmp_path, downloader(b"tampered", []))
    assert list(tmp_path.iterdir()) == []


def test_downloads_need_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="https"):
        kk.download("http://example.test/model.onnx", tmp_path / "x")


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
    assert kk.ipa_to_kokoro(ipa) == kokoro


def test_ipa_to_kokoro_british() -> None:
    assert kk.ipa_to_kokoro("ɡˈə^ʊ", british=True) == "ɡˈQ"


def completed(stdout: str, code: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="")


def test_espeak_cli_phonemises_one_word_per_process() -> None:
    commands: list[list[str]] = []

    def run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        assert kwargs["timeout"] > 0
        assert kwargs["check"] is False
        return completed(" zˈuːɹɪt^ʃ\n")

    fallback = kk.EspeakCli("/usr/bin/espeak-ng", run=run)

    assert fallback(SimpleNamespace(text="Zürich")) == ("zˈuɹɪʧ", 2)
    assert commands == [["/usr/bin/espeak-ng", "-q", "--ipa", "--tie=^", "-v", "en-us", "Zürich"]]
    assert fallback.words == ["Zürich"]


@pytest.mark.parametrize("result", [completed("", 0), completed("zˈuːɹɪk", 1)])
def test_espeak_cli_failure_leaves_the_word_silent(
    result: subprocess.CompletedProcess[str],
) -> None:
    fallback = kk.EspeakCli("espeak-ng", run=lambda *a, **k: result)

    assert fallback(SimpleNamespace(text="Qwxz")) == (None, None)
    assert fallback.words == ["Qwxz"]


def test_cli_mode_only_when_espeak_ng_is_on_path() -> None:
    assert kk.espeak_fallback(lambda name: None) is None
    fallback = kk.espeak_fallback(lambda name: f"/usr/bin/{name}")
    assert isinstance(fallback, kk.EspeakCli)
    assert fallback.executable == "/usr/bin/espeak-ng"


# ── trimming and PCM ────────────────────────────────────────────────────────


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
    assert kk.speech_bounds(levels, threshold=1.0) == bounds


def test_to_pcm16_scales_and_clips() -> None:
    pcm = kk.to_pcm16(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))

    assert np.frombuffer(pcm, dtype="<i2").tolist() == [0, 32767, -32767, 32767, -32767]


# ── provider selection ──────────────────────────────────────────────────────


def test_cpu_provider() -> None:
    assert kk.select_providers("cpu", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CPUExecutionProvider"
    ]


def test_cuda_provider_keeps_the_cpu_as_fallback() -> None:
    assert kk.select_providers("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]) == [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]


def test_cuda_missing_falls_back_to_the_cpu_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=kk.__name__):
        providers = kk.select_providers("cuda", ["CPUExecutionProvider"])

    assert providers == ["CPUExecutionProvider"]
    (record,) = caplog.records
    assert record.levelno == logging.WARNING
    assert "CUDAExecutionProvider" in record.getMessage()


# ── a Kokoro built from fakes ───────────────────────────────────────────────

VOCAB = {"h": 1, "ə": 2, "l": 3, "O": 4}


class FakeSession:
    def __init__(self, path: str, providers: list[str], started: list[str] | None) -> None:
        self.path, self.requested = path, providers
        self.started = providers if started is None else started
        self.inputs: list[dict[str, Any]] = []

    def get_providers(self) -> list[str]:
        return self.started

    def run(self, outputs: None, inputs: dict[str, Any]) -> list[Any]:
        self.inputs.append(inputs)
        # silence, a tone as long as the token count, silence
        tone = 0.5 * np.sin(np.linspace(0, 200, 2400 * inputs["tokens"].shape[1]))
        return [np.concatenate([np.zeros(4800), tone, np.zeros(4800)])[None, :]]


class FakeRuntime:
    def __init__(self, available: list[str], started: list[str] | None = None) -> None:
        self.available, self.started = available, started
        self.sessions: list[FakeSession] = []
        self.preloaded = 0

    def preload_dlls(self) -> None:
        self.preloaded += 1

    def get_available_providers(self) -> list[str]:
        return self.available

    def InferenceSession(self, path: str, providers: list[str]) -> FakeSession:  # noqa: N802
        session = FakeSession(path, providers, self.started)
        self.sessions.append(session)
        return session


def fake_g2p(fallback: kk.EspeakCli | None) -> Callable[[str], tuple[str, Any]]:
    def g2p(text: str) -> tuple[str, Any]:
        tokens = [
            SimpleNamespace(text=w, phonemes=None if w == "Qwxz" else "x") for w in text.split()
        ]
        if fallback is not None:
            for token in tokens:
                if token.phonemes is None:
                    fallback(token)
        return "həlO" * len(tokens), tokens

    return g2p


@pytest.fixture
def weights(tmp_path: Path) -> dict[str, kk.Weight]:
    """Fake weight files already in the cache, with their real names."""
    voices = tmp_path / "voices-v1.0.bin"
    with voices.open("wb") as file:
        np.savez(
            file, af_heart=np.ones((kk.MAX_TOKENS + 1, 1, 256)), am_adam=np.zeros((511, 1, 256))
        )
    (tmp_path / "config.json").write_text(json.dumps({"vocab": VOCAB}), encoding="utf-8")
    (tmp_path / "kokoro-v1.0.onnx").write_bytes(b"onnx")
    return {
        name: weight((tmp_path / name).read_bytes(), name)
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin", "config.json")
    }


def no_download(url: str, target: Path) -> None:
    raise AssertionError(f"unexpected download of {url}")


def make_kokoro(
    directory: Path,
    weights: dict[str, kk.Weight],
    runtime: FakeRuntime | None = None,
    **kwargs: Any,
) -> kk.Kokoro:
    kwargs.setdefault("fallback", None)
    return kk.Kokoro(
        directory,
        runtime=runtime or FakeRuntime(["CPUExecutionProvider"]),
        g2p=fake_g2p,
        weights=weights,
        download=no_download,
        **kwargs,
    )


def test_kokoro_runs_on_the_cpu_by_default(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])

    kokoro = make_kokoro(tmp_path, weights, runtime)

    assert runtime.sessions[0].requested == ["CPUExecutionProvider"]
    assert runtime.sessions[0].path == str(tmp_path / "kokoro-v1.0.onnx")
    assert kokoro.provider == "cpu"
    assert runtime.preloaded == 0


def test_kokoro_on_cuda(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    runtime = FakeRuntime(["CUDAExecutionProvider", "CPUExecutionProvider"])

    kokoro = make_kokoro(tmp_path, weights, runtime, provider="cuda")

    assert kokoro.provider == "cuda"
    assert kokoro.providers == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert runtime.preloaded == 1


def test_kokoro_cuda_that_fails_to_start_uses_the_cpu(
    tmp_path: Path, weights: dict[str, kk.Weight], caplog: pytest.LogCaptureFixture
) -> None:
    runtime = FakeRuntime(
        ["CUDAExecutionProvider", "CPUExecutionProvider"], started=["CPUExecutionProvider"]
    )

    with caplog.at_level(logging.WARNING, logger=kk.__name__):
        kokoro = make_kokoro(tmp_path, weights, runtime, provider="cuda")

    assert kokoro.provider == "cpu"
    assert "could not start CUDAExecutionProvider" in caplog.text


def test_kokoro_cuda_unavailable_uses_the_cpu(
    tmp_path: Path, weights: dict[str, kk.Weight], caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger=kk.__name__):
        kokoro = make_kokoro(tmp_path, weights, provider="cuda")

    assert kokoro.provider == "cpu"
    assert "CUDAExecutionProvider" in caplog.text


def test_kokoro_rejects_an_unknown_voice(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    with pytest.raises(ValueError, match="voice"):
        make_kokoro(tmp_path, weights, voice="xx_nobody")
    kokoro = make_kokoro(tmp_path, weights)
    with pytest.raises(ValueError, match="voice"):
        kokoro.synthesize("Hello", voice="xx_nobody")


def test_synthesis_trims_silence_and_counts_silent_oov(
    tmp_path: Path, weights: dict[str, kk.Weight]
) -> None:
    kokoro = make_kokoro(tmp_path, weights)

    speech = kokoro.synthesize("Hello Qwxz")

    assert kokoro.oov_mode == "silent"
    assert speech.oov_words == ("Qwxz",)
    assert speech.phonemes == "həlOhəlO"
    tokens = kokoro.session.inputs[0]["tokens"]
    assert tokens.tolist() == [[0, 1, 2, 3, 4, 1, 2, 3, 4, 0]]
    assert speech.duration_s < (len(tokens[0]) * 2400 + 9600) / kk.SAMPLE_RATE
    assert speech.duration_s > 0.5


def test_synthesis_counts_oov_words_sent_to_espeak(
    tmp_path: Path, weights: dict[str, kk.Weight]
) -> None:
    fallback = kk.EspeakCli("espeak-ng", run=lambda *a, **k: completed("kwˈɪks"))
    kokoro = make_kokoro(tmp_path, weights, fallback=fallback)

    assert kokoro.oov_mode == "cli"
    assert kokoro.synthesize("Hello Qwxz").oov_words == ("Qwxz",)
    assert kokoro.synthesize("Hello").oov_words == ()


def test_synthesis_refuses_too_many_tokens(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    kokoro = make_kokoro(tmp_path, weights)

    with pytest.raises(ValueError, match="at most 510"):
        kokoro.synthesize(" ".join(["word"] * 130))


def test_trim_keeps_empty_and_all_quiet_audio(
    tmp_path: Path, weights: dict[str, kk.Weight]
) -> None:
    kokoro = make_kokoro(tmp_path, weights)

    assert len(kokoro.trim(np.zeros(0, dtype=np.float32))) == 0
    assert len(kokoro.trim(np.zeros(100, dtype=np.float32))) == 100


# ── the TextToSpeech port ───────────────────────────────────────────────────


async def collect(tts: TextToSpeech, text: str, voice: str = "af_heart") -> list[AudioChunk]:
    return [chunk async for chunk in tts.stream_speech(text, language=EN, voice=voice)]


def test_one_audio_chunk_per_text_chunk(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    tts = kk.KokoroTextToSpeech(make_kokoro(tmp_path, weights))

    first = asyncio.run(collect(tts, "Sure,"))
    second = asyncio.run(collect(tts, "I love that question.", voice="am_adam"))

    assert len(first) == len(second) == 1
    chunk = first[0]
    assert chunk.sample_rate == kk.SAMPLE_RATE
    assert chunk.channels == 1
    assert chunk.duration_s > 0
    assert tts.provider == "cpu"


def test_blank_text_yields_no_audio(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    tts = kk.KokoroTextToSpeech(make_kokoro(tmp_path, weights))

    assert asyncio.run(collect(tts, "  ")) == []


def test_only_english_is_spoken(tmp_path: Path, weights: dict[str, kk.Weight]) -> None:
    tts = kk.KokoroTextToSpeech(make_kokoro(tmp_path, weights))

    async def speak_spanish() -> list[AudioChunk]:
        return [c async for c in tts.stream_speech("Hola", language=LanguageCode("es"), voice="x")]

    with pytest.raises(UnsupportedLanguage):
        asyncio.run(speak_spanish())


def test_empty_synthesis_yields_no_audio(
    tmp_path: Path, weights: dict[str, kk.Weight], monkeypatch: pytest.MonkeyPatch
) -> None:
    kokoro = make_kokoro(tmp_path, weights)
    monkeypatch.setattr(
        kokoro, "synthesize", lambda text, voice=None: kk.Speech(np.zeros(0), "", ())
    )

    assert asyncio.run(collect(kk.KokoroTextToSpeech(kokoro), "Hello")) == []
