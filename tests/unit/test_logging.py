from __future__ import annotations

import logging
from pathlib import Path

from aerovalid.core.logging import setup_logging


def test_setup_logging_attaches_run_id() -> None:
    logger = setup_logging(run_id="run-123", level=logging.INFO)
    assert logging.getLogger("aerovalid") is logger
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


def test_setup_logging_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "run" / "log.txt"
    logger = setup_logging(run_id="run-456", log_file=log_file)
    logger.info("hello")
    for handler in logger.handlers:
        handler.flush()
    assert log_file.exists()
    content = log_file.read_text()
    assert "run-456" in content
    assert "hello" in content


def test_setup_logging_resets_handlers_on_repeat_calls() -> None:
    setup_logging(run_id="first")
    logger = setup_logging(run_id="second")
    assert len(logger.handlers) == 1
