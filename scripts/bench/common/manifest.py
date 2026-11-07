"""The manifest of the synthetic speech fixtures, and the checks on its WAV files.

`fixtures/synthetic/stt/manifest.json` lists every utterance (`id`, `cefr`, `voice`,
`text`) and, once `python -m scripts.bench.fixtures` has generated the audio, a
`generator` block with the Kokoro weight hashes, package versions and OOV mode,
plus each utterance's `duration_s`. Every utterance has `<id>.wav` next to the
manifest: 16 kHz, mono, 16-bit PCM, under 500 KB (the pre-commit large-file limit).
No real voice is ever used.
"""

import re
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2
MAX_BYTES = 500 * 1024
MIN_UTTERANCES = 10
MIN_BANDS = 3
DURATION_S = (3.0, 15.0)
CEFR_BANDS = ("A1", "A2", "B1", "B2", "C1", "C2")
GENERATOR_FIELDS = ("kokoro_model_sha256", "voices_sha256", "packages", "oov_mode")
_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _utterance_problems(index: int, entry: object, seen: set[str]) -> list[str]:
    if not isinstance(entry, Mapping):
        return [f"utterances[{index}] must be an object"]
    problems = []
    uid = entry.get("id")
    if not isinstance(uid, str) or not _ID.match(uid):
        problems.append(f"utterances[{index}].id must be a lowercase slug")
    elif uid in seen:
        problems.append(f"utterances[{index}].id {uid!r} is repeated")
    else:
        seen.add(uid)
    if entry.get("cefr") not in CEFR_BANDS:
        problems.append(f"utterances[{index}].cefr must be one of {', '.join(CEFR_BANDS)}")
    for field in ("text", "voice"):
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            problems.append(f"utterances[{index}].{field} must be a non-empty string")
    return problems


def _wav_problems(path: Path) -> list[str]:
    name = path.name
    problems = []
    if path.stat().st_size >= MAX_BYTES:
        problems.append(f"{name} is {path.stat().st_size} bytes, the limit is {MAX_BYTES}")
    try:
        with wave.open(str(path), "rb") as audio:
            rate, channels, width = audio.getframerate(), audio.getnchannels(), audio.getsampwidth()
            duration = audio.getnframes() / rate
    except (wave.Error, EOFError) as error:
        return [*problems, f"{name} is not a PCM WAV file: {error}"]
    if (rate, channels, width) != (SAMPLE_RATE, 1, SAMPLE_WIDTH):
        problems.append(
            f"{name} is {rate} Hz, {channels} channel(s), {8 * width}-bit; "
            f"expected {SAMPLE_RATE} Hz mono {8 * SAMPLE_WIDTH}-bit"
        )
    low, high = DURATION_S
    if not low <= duration <= high:
        problems.append(f"{name} lasts {duration:.2f} s, expected {low:g}-{high:g} s")
    return problems


def manifest_problems(
    manifest: Mapping[str, Any], directory: Path, *, generated: bool = True
) -> list[str]:
    """Why the manifest or its WAV files break the rules; empty when they are valid.

    With `generated=False` only the text entries are checked (before the audio exists).
    """
    entries = manifest.get("utterances")
    if not isinstance(entries, list):
        return ["utterances must be a list"]
    seen: set[str] = set()
    problems = [p for i, e in enumerate(entries) for p in _utterance_problems(i, e, seen)]
    if len(entries) < MIN_UTTERANCES:
        problems.append(f"{len(entries)} utterances, at least {MIN_UTTERANCES} are required")
    bands = {e.get("cefr") for e in entries if isinstance(e, Mapping)} & set(CEFR_BANDS)
    if len(bands) < MIN_BANDS:
        problems.append(f"{len(bands)} CEFR bands, at least {MIN_BANDS} are required")
    if not generated:
        return problems

    generator = manifest.get("generator")
    if not isinstance(generator, Mapping):
        problems.append("generator must be an object once the audio is generated")
    else:
        problems += [f"generator.{f} is missing" for f in GENERATOR_FIELDS if not generator.get(f)]
    wavs = {p.stem: p for p in directory.glob("*.wav")}
    for uid in sorted(seen):
        if uid not in wavs:
            problems.append(f"{uid}.wav is missing")
        else:
            problems += _wav_problems(wavs[uid])
    problems += [f"{name}.wav has no manifest entry" for name in sorted(set(wavs) - seen)]
    return problems
