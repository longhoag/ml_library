"""Tests for the logging module."""

import logging
import os
import tempfile

import pytest

from ml_library.logging import configure_logging, get_logger


def test_get_logger():
    """Test getting a logger instance."""
    # Test getting a logger with specific name
    logger = get_logger("ml_library")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "ml_library"

    # Test getting a logger with custom name
    custom_logger = get_logger("custom_module")
    assert isinstance(custom_logger, logging.Logger)
    assert custom_logger.name == "custom_module"

    # Test logger different levels
    assert logger != custom_logger


def test_configure_logging():
    """Test setting up logging with various configurations."""
    # Test with default config
    configure_logging()  # This should not raise any exceptions
    # Just verify it doesn't raise exceptions
    _ = get_logger("test_default")

    # Test with custom level
    configure_logging(level=logging.DEBUG)
    # Just verify it doesn't raise exceptions
    _ = get_logger("test_debug")

    # Test with custom level name
    configure_logging(level="warning")
    # Just verify it doesn't raise exceptions
    _ = get_logger("test_warning")


@pytest.mark.parametrize(
    "log_level",
    [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ],
)
def test_log_levels(log_level):
    """Test logging at different levels."""
    configure_logging(level=log_level)
    logger = get_logger("test_levels")

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        # Add a handler to capture logs
        handler = logging.FileHandler(tmp.name)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Log messages at different levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Flush and close
        handler.flush()
        logger.removeHandler(handler)

        # Check the log file content
        with open(tmp.name, "r", encoding="utf-8") as f:
            content = f.read()

            # Check that only messages at or above the configured level were logged
            if log_level <= logging.DEBUG:
                assert "DEBUG:test_levels:Debug message" in content
            if log_level <= logging.INFO:
                assert "INFO:test_levels:Info message" in content
            if log_level <= logging.WARNING:
                assert "WARNING:test_levels:Warning message" in content
            if log_level <= logging.ERROR:
                assert "ERROR:test_levels:Error message" in content
            if log_level <= logging.CRITICAL:
                assert "CRITICAL:test_levels:Critical message" in content

            # Check that messages below the configured level were not logged
            if log_level > logging.DEBUG:
                assert "DEBUG:test_levels:Debug message" not in content
            if log_level > logging.INFO:
                assert "INFO:test_levels:Info message" not in content
            if log_level > logging.WARNING:
                assert "WARNING:test_levels:Warning message" not in content
            if log_level > logging.ERROR:
                assert "ERROR:test_levels:Error message" not in content

    # Clean up
    if os.path.exists(tmp.name):
        os.remove(tmp.name)


def test_logger_setup():
    """Test basic logger setup and usage."""
    # Create a logger
    logger = get_logger("test_setup")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_setup"

    # Configure logging
    configure_logging(level=logging.DEBUG)

    # Verify logger exists
    assert logging.getLogger("test_setup") is logger


class TestLogging:
    """Test logging configuration and functionality."""

    def __init__(self):
        """Initialize test class."""
        self.log_dir = os.path.join(os.path.dirname(__file__), "logs")

    def setup_method(self):
        """Set up test cases."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def test_setup_logging(self):
        """Test logging setup configuration."""
        # Configure logging
        log_file = os.path.join(self.log_dir, "test.log")
        configure_logging(log_file=log_file)

        test_logger = get_logger(__name__)
        assert isinstance(test_logger, logging.Logger)
        assert os.path.exists(log_file)

        # Clean up
        if os.path.exists(log_file):
            os.remove(log_file)
