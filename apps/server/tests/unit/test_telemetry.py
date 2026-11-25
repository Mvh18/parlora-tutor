"""No model library phones home from the server (AGENTS.md: privacy rules)."""

import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_importing_the_server_turns_off_onnxruntime_telemetry() -> None:
    # onnxruntime >= 1.31 sends 1DS telemetry to Microsoft unless this is set before
    # it loads; a fresh interpreter shows what importing the package alone does.
    env = {k: v for k, v in os.environ.items() if k != "ORT_DISABLE_TELEMETRY"}
    code = "import os, parlora_server; print(os.environ.get('ORT_DISABLE_TELEMETRY'))"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "1"


def test_an_explicit_setting_is_kept() -> None:
    env = {**os.environ, "ORT_DISABLE_TELEMETRY": "1"}
    code = "import os, parlora_server; print(os.environ['ORT_DISABLE_TELEMETRY'])"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "1"
