"""Logging configuration for fpspy.

This module provides custom logging formatters and setup functions.
"""

import logging


class WindowIdxFormatter(logging.Formatter):
    """Custom formatter that handles optional window_idx attribute.

    If a log record doesn't have window_idx, it displays '-' instead.
    """

    def format(self, record):
        if not hasattr(record, 'window_idx'):
            record.window_idx = '-'
        return super().format(record)


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging with custom formatter that includes window_idx.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Raises
    ------
    ValueError
        If log_level is not a valid logging level.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    handler = logging.StreamHandler()
    handler.setFormatter(
        WindowIdxFormatter(
            "%(asctime)s - [win:%(window_idx)s] - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logging.basicConfig(
        level=numeric_level,
        handlers=[handler],
    )
