"""Make the NVIDIA CUDA libraries installed as pip wheels loadable on Windows.

On Linux the images set `LD_LIBRARY_PATH`. On Windows, CTranslate2 (faster-whisper)
finds cuBLAS and cuDNN only once each wheel's `nvidia/<library>/bin` directory is
registered as a DLL directory and put on `PATH`. The speech-to-text benchmark uses
this same function.
"""

import os
import site
import sys
from pathlib import Path


def site_packages() -> list[str]:
    return list(site.getsitepackages())


def add_dll_directory(path: str) -> None:
    os.add_dll_directory(path)  # type: ignore[attr-defined,unused-ignore]  # Windows only


def on_windows() -> bool:
    # A call, not an inline check: mypy would mark the Windows branch unreachable
    # when it type-checks on Linux.
    return sys.platform == "win32"


def add_cuda_dll_directories() -> list[str]:
    """On Windows, register every `nvidia/*/bin` directory; returns the library names."""
    added: list[str] = []
    if not on_windows():
        return added
    for base in site_packages():
        for bin_dir in sorted(Path(base, "nvidia").glob("*/bin")):
            add_dll_directory(str(bin_dir))
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(bin_dir.parent.name)
    return added
