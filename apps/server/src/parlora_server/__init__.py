"""Parlora server: the versioned HTTP API in front of the domain core."""

import os
from importlib.metadata import version

# onnxruntime (Kokoro, Silero VAD, Smart Turn) sends 1DS telemetry to Microsoft
# unless this is set before it loads. Nothing in the server may contact a third
# party, so the package turns it off on import; the images also set it.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")

__version__ = version("parlora-server")
