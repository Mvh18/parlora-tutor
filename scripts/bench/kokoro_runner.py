"""The server's Kokoro text-to-speech adapter, with the benchmarks' weight cache.

The synthesis code lives in `parlora_server.adapters.tts.kokoro` (ADR 0004, `cli`
mode) so the benchmarks, the speech fixtures and the server run the same
implementation. This module only adds where the benchmarks keep the weights:
`PARLORA_BENCH_CACHE` (the bench image's volume) or the user cache directory.
"""

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from parlora_server.adapters.tts import kokoro as _adapter
from parlora_server.adapters.tts.kokoro import (
    DEFAULT_VOICE,
    MAX_TOKENS,
    SAMPLE_RATE,
    WEIGHTS,
    EspeakCli,
    Provider,
    Speech,
    Weight,
    WeightError,
    espeak_fallback,
    fetch,
    ipa_to_kokoro,
    sha256_file,
    speech_bounds,
)

__all__ = [
    "CACHE_ENV",
    "DEFAULT_VOICE",
    "MAX_TOKENS",
    "SAMPLE_RATE",
    "WEIGHTS",
    "EspeakCli",
    "Kokoro",
    "Provider",
    "Speech",
    "Weight",
    "WeightError",
    "cache_dir",
    "espeak_fallback",
    "fetch",
    "ipa_to_kokoro",
    "sha256_file",
    "speech_bounds",
]

CACHE_ENV = "PARLORA_BENCH_CACHE"


def cache_dir(env: Mapping[str, str] = os.environ) -> Path:
    """Where weights are kept: `PARLORA_BENCH_CACHE`, else the user cache directory."""
    if env.get(CACHE_ENV):
        return Path(env[CACHE_ENV]) / "kokoro"
    if sys.platform == "win32" and env.get("LOCALAPPDATA"):
        base = Path(env["LOCALAPPDATA"])
    elif env.get("XDG_CACHE_HOME"):
        base = Path(env["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "parlora" / "bench" / "kokoro"


class Kokoro(_adapter.Kokoro):
    """The server adapter's Kokoro, loading its weights from the benchmark cache."""

    def __init__(self, directory: Path | None = None, **kwargs: Any) -> None:
        super().__init__(directory or cache_dir(), **kwargs)
