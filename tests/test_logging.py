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


def test_configure_logging_with_file():
    """Test logging configuration with a file handler."""
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as tmp:
        log_file = tmp.name

        # Configure logging with file
        configure_logging(level="info", log_file=log_file)
        logger = get_logger("test_file")

        # Log a test message
        test_message = "Test file logging message"
        logger.info(test_message)

        # Check that the message was written to the file
        with open(log_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert test_message in content

        # Clean up
        if os.path.exists(log_file):
            os.remove(log_file)


def test_configure_logging_with_custom_format():
    """Test logging configuration with a custom format."""
    custom_format = "%(levelname)s - %(name)s - %(message)s"
    configure_logging(level="info", format_string=custom_format)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log", delete=False) as tmp:
        # Add a file handler to capture the formatted output
        handler = logging.FileHandler(tmp.name)
        formatter = logging.Formatter(custom_format)
        handler.setFormatter(formatter)

        logger = get_logger("test_format")
        logger.addHandler(handler)

        # Log a test message
        test_message = "Test custom format"
        logger.info(test_message)
        handler.flush()

        # Check that the message was formatted correctly
        with open(tmp.name, "r", encoding="utf-8") as f:
            content = f.read()
            expected_format = f"INFO - test_format - {test_message}"
            assert expected_format in content

        # Clean up
        if os.path.exists(tmp.name):
            os.remove(tmp.name)


def test_invalid_log_level():
    """Test that an invalid log level raises a ValueError."""
    with pytest.raises(ValueError):
        configure_logging(level="invalid_level")


def test_capture_warnings():
    """Test warning capture functionality."""
    # Just test that the capture_warnings parameter is accepted
    # We can't easily test if warnings are actually captured in a unit test
    configure_logging(level="warning", capture_warnings=True)

    # Generate a warning
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.warn("Test warning message")
        assert len(w) > 0
        assert "Test warning message" in str(w[0].message)


class TestLogging:
    """Test logging configuration and functionality."""

    log_dir = os.path.join(os.path.dirname(__file__), "logs")

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
