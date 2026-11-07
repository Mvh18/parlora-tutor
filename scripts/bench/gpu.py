"""Sample the used VRAM of GPU 0 while a benchmark runs.

NVML (`nvidia-ml-py`, in the bench group) is used when installed; otherwise
`nvidia-smi` is polled. Values are MiB for the whole GPU, so other processes count
too: benchmarks record the baseline before loading a model and subtract it.
"""

import shutil
import subprocess
import threading
from types import TracebackType
from typing import Any, Self


class VramSampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.peak_mb: int | None = None
        self._nvml: Any = None
        self._handle: Any = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:  # no bench group, no driver or no GPU: fall back
            self._nvml = None
        self._smi = shutil.which("nvidia-smi")

    @property
    def source(self) -> str | None:
        if self._nvml is not None:
            return "nvml"
        return "nvidia-smi" if self._smi else None

    def used_mb(self) -> int | None:
        if self._nvml is not None:
            return int(self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used) // (1024 * 1024)
        if not self._smi:
            return None
        result = subprocess.run(
            [self._smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        first = result.stdout.strip().splitlines()[:1]
        return int(first[0]) if result.returncode == 0 and first and first[0].isdigit() else None

    def _sample(self) -> None:
        while not self._stop.is_set():
            used = self.used_mb()
            if used is not None and (self.peak_mb is None or used > self.peak_mb):
                self.peak_mb = used
            self._stop.wait(self.interval_s)

    def __enter__(self) -> Self:
        self.peak_mb = None
        self._stop.clear()
        if self.source is not None:
            self._thread = threading.Thread(target=self._sample, daemon=True)
            self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
