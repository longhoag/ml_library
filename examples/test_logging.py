"""Utility script to test logging configuration."""

import argparse

import numpy as np

from ml_library.exceptions import (
    DataError,
    InvalidParameterError,
    MLLibraryError,
    ModelError,
    NotFittedError,
)
from ml_library.logging import configure_logging, get_logger
from ml_library.models import LinearModel

# Get a logger for this script
logger = get_logger(__name__)


def test_logging(level="info", log_file=None):
    """Test logging functionality at different levels.

    Parameters
    ----------
    level : str
        Logging level to test.
    log_file : str, optional
        File to log to.
    """
    # Configure logging
    configure_logging(level=level, log_file=log_file)

    # Output messages at different levels
    logger.debug("This is a DEBUG message")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")
    logger.critical("This is a CRITICAL message")

    # Test with different modules
    model_logger = get_logger("ml_library.models")
    model_logger.info("This is a message from the models module")

    # Log an exception
    try:
        raise ValueError("This is a test exception")
    except ValueError as e:
        logger.exception("Caught an exception: %s", str(e))


def test_error_handling():
    """Test error handling with custom exceptions."""
    try:
        # Test MLLibraryError (base error)
        logger.info("Testing MLLibraryError...")
        raise MLLibraryError("This is a general library error")
    except MLLibraryError as e:
        logger.exception("Caught MLLibraryError: %s", str(e))

    try:
        # Test DataError
        logger.info("Testing DataError...")
        raise DataError("Invalid data format", data_shape=(100, 5))
    except DataError as e:
        logger.exception("Caught DataError: %s", str(e))

    try:
        # Test ModelError
        logger.info("Testing ModelError...")
        raise ModelError(
            "Failed to fit model", model_type="LinearModel", details={"epochs": 100}
        )
    except ModelError as e:
        logger.exception("Caught ModelError: %s", str(e))

    try:
        # Test NotFittedError
        logger.info("Testing NotFittedError...")
        model = LinearModel()
        # Should raise NotFittedError
        model.predict(np.array([[1, 2, 3]]))
    except NotFittedError as e:
        logger.exception("Caught NotFittedError: %s", str(e))

    try:
        # Test InvalidParameterError
        logger.info("Testing InvalidParameterError...")
        raise InvalidParameterError(
            "n_estimators", -10, allowed_values="positive integers"
        )
    except InvalidParameterError as e:
        logger.exception("Caught InvalidParameterError: %s", str(e))


def main():
    """Main function to test logging."""
    parser = argparse.ArgumentParser(description="Test ML Library logging")
    parser.add_argument(
        "--level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )
    parser.add_argument("--log-file", help="Log to this file as well as the console")
    args = parser.parse_args()

    print(f"Testing logging at level: {args.level}")
    print(f"Log file: {args.log_file or 'None - console only'}")
    print("-" * 50)

    test_logging(args.level, args.log_file)
    test_error_handling()

    print("-" * 50)
    print("Testing complete. Check the logs for details.")


if __name__ == "__main__":
    main()
