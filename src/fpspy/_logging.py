"""Logging configuration for fpspy.

This module provides custom logging formatters and setup functions.
"""

import logging
from pathlib import Path
import time
import rich
import rich.logging
import rich.highlighter
import rich.console


class Highlighter(rich.highlighter.ReprHighlighter):
    def __init__(self):
        super().__init__()
        self.highlights.append(
            # Matches (<path>)       (a path in brackets)
            # The default path regex assumes starting '/'.
            r"\((?P<path>((([-\w._+]+)*)\/[\w._+-]*)*)\)"
        )


_console = rich.console.Console(highlighter=Highlighter())


class WindowIdxFormatter(logging.Formatter):
    """Custom formatter that handles optional window_idx attribute.

    If a log record doesn't have window_idx, it displays '-' instead.
    """

    def format(self, record):
        if not hasattr(record, "window_idx"):
            record.process_name = "main "
        else:
            record.process_name = f"win {record.window_idx}"
        return super().format(record)


def setup_main_logging(log_level: str = "INFO", use_rich=True, show_time=False) -> None:
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
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    root_logger.setLevel(numeric_level)

    if use_rich:
        handler = rich.logging.RichHandler(
            console=_console,
            show_time=show_time,
            omit_repeated_times=False,
            show_level=True,
            show_path=True,
            rich_tracebacks=True,
        )
        handler.setFormatter(WindowIdxFormatter("%(process_name)s | %(message)s"))
    else:
        handler = logging.StreamHandler()
        formatter = WindowIdxFormatter(
            "%(asctime)s | %(process_name)s | [%(levelname)8s]: %(message)s"
        )
        handler.setFormatter(formatter)
    root_logger.addHandler(handler)


class WithCloseMsgStreamHandler(logging.StreamHandler):
    def __init__(self, close_msg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.close_msg = close_msg

    def close(self):
        """
        See: https://github.com/python/cpython/blob/3.13/Lib/logging/__init__.py
        """
        self.emit(
            logging.makeLogRecord(
                {
                    "msg": self.close_msg,
                    "args": (),
                    "levelname": "INFO",
                    "levelno": logging.INFO,
                }
            )
        )
        super().close()


def enable_file_logging(log_path):
    log_path = Path(log_path)
    root_logger = logging.getLogger()
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ | %(filename)12s:%(lineno)d [%(levelname)8s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime  # Use UTC time
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    log_path_msg = f"{log_path} [log path]"
    logging.info(log_path_msg)
    # Handler to print the log path when logging finishes.
    end_handler = WithCloseMsgStreamHandler(close_msg=log_path_msg)
    # Don't accept any logs from this handler.
    end_handler.addFilter(lambda x: False)
    # The closing message isn't working, but not sure why:
    root_logger.addHandler(end_handler)
