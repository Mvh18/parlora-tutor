"""Log levels of the server and of Pipecat's loguru sink."""

import logging
from collections.abc import Iterator

import pytest
from loguru import logger as loguru_logger

from parlora_server.config import Settings
from parlora_server.logs import configure_logging

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def restore_server_logger() -> Iterator[None]:
    server = logging.getLogger("parlora_server")
    handlers, level, propagate = list(server.handlers), server.level, server.propagate
    yield
    server.handlers[:] = handlers
    server.setLevel(level)
    server.propagate = propagate
    loguru_logger.remove()
    loguru_logger.add(__import__("sys").stderr)


def test_info_is_the_default_level() -> None:
    assert Settings(_env_file=None).log_level == "INFO"


def test_server_info_records_are_kept_and_debug_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    log = logging.getLogger("parlora_server.voice.metrics")

    log.info("turn voice_to_voice_ms=900")
    log.debug("turn text learner='hello'")

    err = capsys.readouterr().err
    assert "voice_to_voice_ms=900" in err
    assert "hello" not in err


def test_pipecat_debug_lines_are_dropped_at_info(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    loguru_logger.debug("Transcription: hello there")
    loguru_logger.info("pipeline started")

    err = capsys.readouterr().err
    assert "hello there" not in err
    assert "pipeline started" in err


def test_configuring_twice_adds_one_handler() -> None:
    configure_logging("DEBUG")
    configure_logging("DEBUG")

    server = logging.getLogger("parlora_server")
    assert [h.get_name() for h in server.handlers].count("parlora") == 1
    assert server.level == logging.DEBUG
