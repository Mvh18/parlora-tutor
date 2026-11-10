"""Kokoro text-to-speech adapter, GPL-free (ADR 0004, `cli` mode).

misaki's English G2P (without extras) turns text into Kokoro phonemes. Words
missing from its dictionary (out-of-vocabulary, OOV) go to a separately installed
`espeak-ng` executable, one word per subprocess, when one is on `PATH` (`cli`
mode); otherwise they stay silent and are counted (`silent` mode). No GPL code is
imported. The phonemes, a voice style vector and the speed go into the Kokoro ONNX
model on onnxruntime's CPU or CUDA provider (`PARLORA_TTS__PROVIDER`); when CUDA is
requested but onnxruntime cannot use it, the CPU is used and a warning is logged.
Leading and trailing silence is trimmed.

The benchmarks (`scripts/bench/kokoro_runner.py`) and the speech fixtures use this
same code.

Adapted code, with thanks:
- kokoro-onnx by thewh1teagle, MIT License
  (https://github.com/thewh1teagle/kokoro-onnx): the ONNX input layout, the style
  vector chosen by token count and the 510-token limit.
- misaki by hexgrad, Apache License 2.0 (https://github.com/hexgrad/misaki): the
  table and rules in `ipa_to_kokoro` that map espeak-ng IPA to Kokoro phonemes,
  from `misaki/espeak.py`.

Weights (Kokoro-82M, Apache-2.0) are downloaded on first use into the directory
the caller gives (the models volume in the server image) and verified by SHA-256;
they are never stored in the repository.
"""

import asyncio
import hashlib
import json
import logging
import re
import shutil
import subprocess
import threading
import urllib.request
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from parlora_core.audio import AudioChunk
from parlora_core.errors import UnsupportedLanguage
from parlora_core.languages import LanguageCode

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24_000
MAX_TOKENS = 510
DEFAULT_VOICE = "af_heart"
ENGLISH = LanguageCode("en")

Provider = Literal["cpu", "cuda"]
CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"


@dataclass(frozen=True)
class Weight:
    name: str
    url: str
    sha256: str


_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_HF_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
WEIGHTS: Mapping[str, Weight] = {
    w.name: w
    for w in (
        Weight(
            "kokoro-v1.0.onnx",
            f"{_RELEASE}/kokoro-v1.0.onnx",
            "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
        ),
        Weight(
            "voices-v1.0.bin",
            f"{_RELEASE}/voices-v1.0.bin",
            "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
        ),
        # The phoneme vocabulary, from the Kokoro-82M repository at a fixed revision.
        Weight(
            "config.json",
            f"https://huggingface.co/hexgrad/Kokoro-82M/resolve/{_HF_REVISION}/config.json",
            "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
        ),
    )
}


class WeightError(RuntimeError):
    """A downloaded weight file does not match its pinned SHA-256."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if not url.startswith("https://"):
        raise ValueError(f"weights are only downloaded over https, got {url!r}")
    with urllib.request.urlopen(url, timeout=300) as response, target.open("wb") as file:  # noqa: S310 - https checked above
        shutil.copyfileobj(response, file)


def fetch(
    weight: Weight, directory: Path, download: Callable[[str, Path], None] = download
) -> Path:
    """The verified local copy of `weight`, downloading it when missing or corrupt."""
    path = directory / weight.name
    if path.is_file() and sha256_file(path) == weight.sha256:
        return path
    directory.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    download(weight.url, partial)
    actual = sha256_file(partial)
    if actual != weight.sha256:
        partial.unlink()
        raise WeightError(f"{weight.name}: SHA-256 {actual}, expected {weight.sha256}")
    partial.replace(path)
    return path


# ── onnxruntime provider ────────────────────────────────────────────────────


def select_providers(requested: Provider, available: Sequence[str]) -> list[str]:
    """The onnxruntime providers for `requested`, falling back to the CPU with a warning."""
    if requested == "cuda":
        if CUDA_PROVIDER in available:
            return [CUDA_PROVIDER, CPU_PROVIDER]
        logger.warning(
            "Kokoro: %s requested but onnxruntime does not offer it; using %s",
            CUDA_PROVIDER,
            CPU_PROVIDER,
        )
    return [CPU_PROVIDER]


# ── OOV words through the espeak-ng executable ──────────────────────────────

# espeak-ng IPA -> Kokoro phonemes, longest first (misaki/espeak.py, Apache-2.0).
_E2M = sorted(
    {
        "ʔˌn̩": "ʔn",
        "ʔn̩": "ʔn",
        "a^ɪ": "I",
        "a^ʊ": "W",
        "d^ʒ": "ʤ",
        "e^ɪ": "A",
        "e": "A",
        "t^ʃ": "ʧ",
        "ɔ^ɪ": "Y",
        "ə^l": "ᵊl",
        "ʲo": "jo",
        "ʲə": "jə",
        "ʲ": "",
        "ɚ": "əɹ",
        "r": "ɹ",
        "x": "k",
        "ç": "k",
        "ɐ": "ə",
        "ɬ": "l",
        "̃": "",
    }.items(),
    key=lambda pair: -len(pair[0]),
)


def ipa_to_kokoro(ipa: str, british: bool = False) -> str:
    """Map espeak-ng IPA (stress marks, `^` ties) to Kokoro's phoneme set."""
    ps = ipa.strip()
    for old, new in _E2M:
        ps = ps.replace(old, new)
    ps = re.sub("(\\S)̩", "ᵊ\\1", ps).replace("̩", "")
    if british:
        ps = ps.replace("e^ə", "ɛː").replace("iə", "ɪə").replace("ə^ʊ", "Q")
    else:
        ps = ps.replace("o^ʊ", "O").replace("ɜːɹ", "ɜɹ").replace("ɜː", "ɜɹ")
        ps = ps.replace("ɪə", "iə").replace("ː", "")
    ps = ps.replace("o", "ɔ")  # espeak-ng before 1.52
    ps = ps.replace("ɾ", "T").replace("ʔ", "t")
    return ps.replace("^", "")


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass
class EspeakCli:
    """misaki fallback: one OOV word per `espeak-ng` process, counted."""

    executable: str
    british: bool = False
    run: Runner = subprocess.run
    words: list[str] = field(default_factory=list)

    def __call__(self, token: Any) -> tuple[str | None, int | None]:
        text = str(token.text)
        self.words.append(text)
        voice = "en-gb" if self.british else "en-us"
        result = self.run(
            [self.executable, "-q", "--ipa", "--tie=^", "-v", voice, text],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        ipa = result.stdout.strip() if result.returncode == 0 else ""
        return (ipa_to_kokoro(ipa, self.british), 2) if ipa else (None, None)


def espeak_fallback(which: Callable[[str], str | None] = shutil.which) -> EspeakCli | None:
    """`cli` mode when an `espeak-ng` executable is on `PATH`, else `None` (`silent`)."""
    executable = which("espeak-ng")
    return EspeakCli(executable) if executable else None


# ── Silence trimming and PCM ────────────────────────────────────────────────


def speech_bounds(levels: Sequence[float], threshold: float) -> tuple[int, int]:
    """First and one-past-last frame whose level reaches `threshold`; (0, 0) if none."""
    loud = [i for i, level in enumerate(levels) if level >= threshold]
    return (loud[0], loud[-1] + 1) if loud else (0, 0)


def to_pcm16(audio: Any) -> bytes:
    """Float audio in [-1, 1] as 16-bit little-endian PCM, clipping what lies outside."""
    import numpy as np

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return bytes((clipped * 32767).astype("<i2").tobytes())


# ── Synthesis ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Speech:
    audio: Any  # float32 NumPy array at SAMPLE_RATE
    phonemes: str
    oov_words: tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return len(self.audio) / SAMPLE_RATE


def _misaki_g2p(fallback: EspeakCli | None) -> Callable[[str], tuple[str, Any]]:
    from misaki import en

    g2p: Callable[[str], tuple[str, Any]] = en.G2P(
        trf=False, british=False, fallback=fallback, unk=""
    )
    return g2p


class Kokoro:
    """Text to 24 kHz float32 audio on onnxruntime's CPU or CUDA provider.

    `runtime` (the onnxruntime module), `g2p`, `weights` and `download` exist so the
    adapter can be tested without the real model.
    """

    def __init__(
        self,
        directory: Path,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        fallback: EspeakCli | str | None = "auto",
        provider: Provider = "cpu",
        *,
        runtime: ModuleType | Any = None,
        g2p: Callable[[EspeakCli | None], Callable[[str], tuple[str, Any]]] = _misaki_g2p,
        weights: Mapping[str, Weight] = WEIGHTS,
        download: Callable[[str, Path], None] = download,
    ) -> None:
        import numpy as np

        if runtime is None:
            import onnxruntime

            runtime = onnxruntime

        paths = {name: fetch(weight, directory, download) for name, weight in weights.items()}
        self._np = np
        self.voice, self.speed = voice, speed
        self.fallback: EspeakCli | None = (
            espeak_fallback()
            if fallback == "auto"
            else fallback
            if isinstance(fallback, EspeakCli)
            else None
        )
        self.oov_mode = "cli" if self.fallback is not None else "silent"
        self.vocab: dict[str, int] = json.loads(paths["config.json"].read_text(encoding="utf-8"))[
            "vocab"
        ]
        self.voices = np.load(paths["voices-v1.0.bin"])
        if voice not in self.voices.files:
            raise ValueError(f"unknown Kokoro voice {voice!r}")
        providers = select_providers(provider, runtime.get_available_providers())
        if CUDA_PROVIDER in providers and hasattr(runtime, "preload_dlls"):
            # onnxruntime-gpu finds the CUDA and cuDNN libraries installed as wheels
            # only once they are preloaded (onnxruntime >= 1.21).
            runtime.preload_dlls()
        self.session = runtime.InferenceSession(str(paths["kokoro-v1.0.onnx"]), providers=providers)
        self.providers: tuple[str, ...] = tuple(self.session.get_providers())
        self.provider: Provider = "cuda" if CUDA_PROVIDER in self.providers else "cpu"
        if CUDA_PROVIDER in providers and self.provider != "cuda":
            logger.warning(
                "Kokoro: onnxruntime could not start %s; using %s", CUDA_PROVIDER, CPU_PROVIDER
            )
        self._g2p = g2p(self.fallback)
        self.hashes = {name: weight.sha256 for name, weight in weights.items()}

    def phonemize(self, text: str) -> tuple[str, tuple[str, ...]]:
        before = len(self.fallback.words) if self.fallback else 0
        phonemes, tokens = self._g2p(text)
        if self.fallback is not None:
            oov = tuple(self.fallback.words[before:])
        else:
            oov = tuple(
                t.text for t in tokens if t.phonemes is None and any(c.isalnum() for c in t.text)
            )
        return phonemes, oov

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        np = self._np
        voice = voice or self.voice
        if voice not in self.voices.files:
            raise ValueError(f"unknown Kokoro voice {voice!r}")
        phonemes, oov = self.phonemize(text)
        ids = [self.vocab[p] for p in phonemes if p in self.vocab]
        if len(ids) > MAX_TOKENS:
            raise ValueError(f"{len(ids)} phoneme tokens, Kokoro takes at most {MAX_TOKENS}")
        inputs = {
            "tokens": np.array([[0, *ids, 0]], dtype=np.int64),
            "style": np.asarray(self.voices[voice][len(ids)], dtype=np.float32),
            "speed": np.array([self.speed], dtype=np.float32),
        }
        audio = self.session.run(None, inputs)[0].astype(np.float32).reshape(-1)
        return Speech(self.trim(audio), phonemes, oov)

    def trim(self, audio: Any, frame: int = 240, threshold_db: float = -45.0) -> Any:
        """Drop leading and trailing frames quieter than `threshold_db` below the peak."""
        np = self._np
        if not len(audio):
            return audio
        frames = len(audio) // frame
        rms = np.sqrt(np.mean(np.square(audio[: frames * frame].reshape(frames, frame)), axis=1))
        peak = float(rms.max()) if frames else 0.0
        start, end = speech_bounds(rms.tolist(), peak * 10 ** (threshold_db / 20))
        return audio[start * frame : end * frame] if end else audio


# ── The TextToSpeech port ───────────────────────────────────────────────────


class KokoroTextToSpeech:
    """`parlora_core.ports.TextToSpeech` over Kokoro: one audio chunk per text chunk.

    The caller (the voice pipeline) already cuts the reply into clauses, so each call
    synthesises the text it is given in one piece, off the event loop. Calls are
    serialised: misaki and the OOV counter are not thread-safe.
    """

    def __init__(self, kokoro: Kokoro) -> None:
        self._kokoro = kokoro
        self._lock = threading.Lock()

    @property
    def provider(self) -> Provider:
        return self._kokoro.provider

    async def stream_speech(
        self, text: str, *, language: LanguageCode, voice: str
    ) -> AsyncIterator[AudioChunk]:
        if language != ENGLISH:
            raise UnsupportedLanguage(f"Kokoro speaks only {ENGLISH}, not {language}")
        if not text.strip():
            return
        speech = await asyncio.to_thread(self._synthesize, text, voice)
        if len(speech.audio):
            yield AudioChunk(to_pcm16(speech.audio), SAMPLE_RATE)

    def _synthesize(self, text: str, voice: str) -> Speech:
        with self._lock:
            return self._kokoro.synthesize(text, voice)
