"""Generate the synthetic speech fixtures from their committed manifest.

Each utterance in `fixtures/synthetic/stt/manifest.json` is synthesised with the
benchmark's Kokoro runner, resampled from 24 kHz to 16 kHz, peak-normalised and
written as 16-bit mono PCM `<id>.wav` next to the manifest. The manifest is then
filled with the generator facts (weight hashes, package and espeak-ng versions, OOV
mode) and each utterance's duration and OOV words, so a regenerated set is
traceable. Run it in the bench image (`just bench-fixtures`), where espeak-ng gives
OOV words their `cli` pronunciation; elsewhere it refuses unless `--allow-silent`.

Usage: python -m scripts.bench.fixtures [--allow-silent]
"""

import argparse
import json
import shutil
import subprocess
import sys
import wave
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from scripts.bench.common.manifest import SAMPLE_RATE, manifest_problems
from scripts.bench.kokoro_runner import SAMPLE_RATE as KOKORO_RATE
from scripts.bench.kokoro_runner import Kokoro

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "synthetic" / "stt"
PEAK = 10 ** (-1 / 20)  # -1 dBFS
PACKAGES = ("misaki", "onnxruntime", "spacy", "en-core-web-sm", "numpy")


def resample_24k_to_16k(audio: Any) -> Any:
    """Rational 2/3 resampling: upsample by 2, low-pass at 7.6 kHz, keep every 3rd."""
    import numpy as np

    taps = 241
    up = 2 * KOKORO_RATE
    n = np.arange(taps) - (taps - 1) / 2
    cutoff = 7_600 / up
    kernel = 2 * cutoff * np.sinc(2 * cutoff * n) * np.kaiser(taps, 8.6)
    kernel *= 2 / kernel.sum()  # unity gain after inserting zeros
    stuffed = np.zeros(len(audio) * 2, dtype=np.float64)
    stuffed[::2] = audio
    return np.convolve(stuffed, kernel, mode="same")[::3].astype(np.float32)


def to_pcm16(audio: Any) -> bytes:
    import numpy as np

    peak = float(np.max(np.abs(audio))) or 1.0
    scaled = np.clip(audio * (PEAK / peak), -1.0, 1.0)
    return bytes((scaled * 32767).astype("<i2").tobytes())


def write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm)


def espeak_version() -> str | None:
    executable = shutil.which("espeak-ng")
    if not executable:
        return None
    out = subprocess.run(
        [executable, "--version"], capture_output=True, text=True, check=False, timeout=10
    ).stdout
    parts = out.split()
    return parts[parts.index("NG") + 2].rstrip(":") if "NG" in parts[:-2] else out.strip()


def generate(directory: Path, allow_silent: bool) -> int:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = manifest_problems(manifest, directory, generated=False)
    if problems:
        print("\n".join(f"manifest: {p}" for p in problems), file=sys.stderr)
        return 1
    kokoro = Kokoro()
    if kokoro.oov_mode != "cli" and not allow_silent:
        print(
            "fixtures: no espeak-ng on PATH, OOV words would be silent; run "
            "`just bench-fixtures` (bench image) or pass --allow-silent",
            file=sys.stderr,
        )
        return 1
    for stale in directory.glob("*.wav"):
        stale.unlink()
    for entry in manifest["utterances"]:
        speech = kokoro.synthesize(entry["text"], voice=entry["voice"])
        audio = resample_24k_to_16k(speech.audio)
        write_wav(directory / f"{entry['id']}.wav", to_pcm16(audio))
        entry["duration_s"] = round(len(audio) / SAMPLE_RATE, 2)
        entry["oov_words"] = list(speech.oov_words)
        print(f"fixtures: {entry['id']} {entry['duration_s']} s, OOV {speech.oov_words}")
    manifest["generator"] = {
        "script": "scripts/bench/fixtures.py",
        "kokoro_model_sha256": kokoro.hashes["kokoro-v1.0.onnx"],
        "voices_sha256": kokoro.hashes["voices-v1.0.bin"],
        "config_sha256": kokoro.hashes["config.json"],
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "espeak_ng": espeak_version(),
        "oov_mode": kokoro.oov_mode,
        "speed": kokoro.speed,
        "sample_rate": SAMPLE_RATE,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    problems = manifest_problems(manifest, directory)
    print("\n".join(f"fixtures: {p}" for p in problems), file=sys.stderr)
    return 1 if problems else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--allow-silent", action="store_true")
    parser.add_argument("--directory", type=Path, default=FIXTURES)
    args = parser.parse_args(argv)
    return generate(args.directory, args.allow_silent)


if __name__ == "__main__":
    sys.exit(main())
