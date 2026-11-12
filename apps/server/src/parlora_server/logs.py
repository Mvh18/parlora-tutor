"""Log levels of the server and of Pipecat (design D8).

Uvicorn configures only its own loggers, so `parlora_server` records below
WARNING would be lost; Pipecat logs through loguru, whose default sink prints
DEBUG, which includes transcripts. `configure_logging` sends both to stderr at
`PARLORA_LOG_LEVEL` (INFO by default), so turn text appears only when DEBUG is
chosen on purpose. The `parlora_server` logger keeps propagating: under uvicorn
the root logger has no handler, so nothing is printed twice.
"""

import logging
import sys

from loguru import logger as loguru_logger

FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_NAME = "parlora"


def configure_logging(level: str) -> None:
    """Set the level of the `parlora_server` logger and of Pipecat's loguru sink."""
    server = logging.getLogger("parlora_server")
    server.setLevel(level)
    for old in [h for h in server.handlers if h.get_name() == _HANDLER_NAME]:
        server.removeHandler(old)  # a new app (tests) writes to the current stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(FORMAT))
    server.addHandler(handler)

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level=level)
