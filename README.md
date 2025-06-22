# ML Library

A production-ready machine learning library designed for flexibility, extensibility, and ease of use.

## Overview

This library provides a comprehensive set of tools for machine learning workflows, including:

- Data preprocessing and feature engineering
- Model training and evaluation
- Model serialization and management
- Visualization utilities
- Integration with popular ML frameworks
- Extensive documentation and examples
- Robust logging and error handling

## Installation

```bash
pip install ml-library
```

## Quick Start

```python
from ml_library import Model, Preprocessor, configure_logging, get_logger

# Configure logging
configure_logging(level="info", log_file="ml_library.log")
logger = get_logger(__name__)

try:
    # Prepare data
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train model
    model = Model()
    model.train(X_train_processed, y_train)

    # Evaluate
    score = model.evaluate(X_test_processed, y_test)
    logger.info("Model accuracy: %.4f", score)

    # Save model
    model.save("model.pkl")
    logger.info("Model saved to model.pkl")
    
except Exception as e:
    logger.exception("Error in ML workflow: %s", str(e))
    # Handle the error appropriately
```

## Documentation

The library comes with comprehensive documentation built with Sphinx. To build and view the documentation:

```bash
# Build the documentation
./build_docs.sh

# Alternatively, build manually
cd docs
make html
```

The documentation will be available in the `docs/build/html/` directory.

### Documentation Structure

- **Introduction**: Overview of the library and its features
- **Installation**: Detailed installation instructions
- **Quickstart**: Quick examples to get started
- **Tutorials**: In-depth tutorials for various use cases
- **API Reference**: Detailed documentation of all modules, classes, and functions
- **Examples**: Complete working examples
- **Contributing**: Guidelines for contributing to the library
- **Changelog**: Version history and changes

## Testing

The library has a comprehensive test suite using pytest. To run the tests:

```bash
# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=ml_library
```

## Logging and Error Handling

The library includes a robust logging and error handling framework:

```python
from ml_library import configure_logging, get_logger
from ml_library.exceptions import DataError

# Configure logging
configure_logging(level="info", log_file="app.log")
logger = get_logger(__name__)

try:
    # Your code here
    logger.info("Processing data with %d features", n_features)
    if some_problem:
        raise DataError("Invalid data format", data_shape=(n_samples, n_features))
except Exception as e:
    logger.exception("Error: %s", str(e))
```

### Key Features

- **Configurable Logging**: Set log level, format, and output destinations
- **Custom Exceptions**: Hierarchical exception system for clear error reporting
- **Consistent Interface**: All components use the same logging and error handling patterns
- **Production Ready**: Designed for use in production environments with proper error recovery

# Run tests with coverage report
python -m pytest --cov=ml_library

# Generate HTML coverage report
python -m pytest --cov=ml_library --cov-report=html
```

For more details on testing, see [tests/README.md](tests/README.md).

## Documentation

Full documentation is available at [docs link].

## Contributing

Contributions are welcome! Please check our contribution guidelines before submitting a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
