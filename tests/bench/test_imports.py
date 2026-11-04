"""The pure benchmark modules import without the optional `bench` group."""

import subprocess
import sys

import pytest

from tests.repo.helpers import REPO_ROOT

pytestmark = pytest.mark.unit

PURE_MODULES = [
    "scripts.bench.common.budgets",
    "scripts.bench.common.machine",
    "scripts.bench.common.privacy",
    "scripts.bench.common.results",
    "scripts.bench.common.schema",
    "scripts.bench.common.stats",
    "scripts.bench.common.wer",
]
HEAVY = ["numpy", "faster_whisper", "ctranslate2", "onnxruntime", "misaki", "pynvml", "httpx"]


def test_pure_modules_load_no_heavy_library() -> None:
    code = (
        "import importlib, sys\n"
        f"for name in {PURE_MODULES!r}: importlib.import_module(name)\n"
        f"print(','.join(m for m in {HEAVY!r} if m in sys.modules))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    assert loaded == "", f"pure benchmark modules imported {loaded}"
