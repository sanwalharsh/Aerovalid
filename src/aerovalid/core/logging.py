"""Structured logging setup. No module uses `print`; every logger goes through
`logging.getLogger(__name__)` and run_id is injected via a filter (CLAUDE.md §7).
"""

from __future__ import annotations

import logging
from pathlib import Path


class RunIdFilter(logging.Filter):
    """Injects a `run_id` attribute into every log record for this process."""

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self._run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = self._run_id
        return True


_LOG_FORMAT = "%(asctime)s %(levelname)-8s [%(run_id)s] %(name)s: %(message)s"


def setup_logging(
    run_id: str = "-",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """
    Configure the root AeroValidate logger.

    Parameters
    ----------
    run_id : str
        Identifier attached to every record, typically the CLI step's run_id.
    level : int
        Logging level, e.g. `logging.INFO`.
    log_file : Path, optional
        If given, also write records to this file (created with parents).

    Returns
    -------
    logging.Logger
        The configured `aerovalid` logger.
    """
    logger = logging.getLogger("aerovalid")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.filters.clear()

    formatter = logging.Formatter(_LOG_FORMAT)
    run_filter = RunIdFilter(run_id)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(run_filter)
    logger.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(run_filter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
