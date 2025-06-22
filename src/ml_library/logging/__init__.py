"""Logging utilities for ML Library."""

import logging
import os
import sys
from typing import Optional, Union

__all__ = ["get_logger", "configure_logging"]

# Default logging format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Dictionary mapping string log levels to actual log levels
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name.

    Parameters
    ----------
    name : str
        The name of the logger, usually __name__ of the calling module.

    Returns
    -------
    logging.Logger
        A configured logger instance.
    """
    return logging.getLogger(name)


def configure_logging(
    level: Union[str, int] = "info",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    capture_warnings: bool = True,
) -> None:
    """Configure the logging for the ML Library.

    Parameters
    ----------
    level : str or int, default="info"
        The logging level. Can be a string ('debug', 'info', 'warning', 'error',
        'critical') or an integer constant from the logging module.
    log_file : str, optional
        If provided, logging will also be directed to this file.
    format_string : str, optional
        Custom formatting string for log messages. If None, the default format is used.
    capture_warnings : bool, default=True
        Whether to capture warnings from the warnings module.

    Returns
    -------
    None
    """
    # Convert string level to logging constant if needed
    if isinstance(level, str):
        level = level.lower()
        numeric_level = LOG_LEVELS.get(level)
        if numeric_level is None:
            raise ValueError(f"Invalid log level: {level}")
        level = numeric_level

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter
    formatter = logging.Formatter(format_string or DEFAULT_FORMAT)
    console_handler.setFormatter(formatter)

    # Add console handler to root logger
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure warning capture
    logging.captureWarnings(capture_warnings)

    # Add a logger for the ml_library package
    ml_logger = logging.getLogger("ml_library")
    ml_logger.info(
        "ML Library logging configured with level: %s", logging.getLevelName(level)
    )


# Configure basic logging by default
configure_logging(level="info")
