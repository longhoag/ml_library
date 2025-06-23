# ML Library

A production-ready machine learning library designed for flexibility, extensibility, and ease of use with comprehensive test coverage and strong type safety.

## Overview

This library provides a comprehensive set of tools for machine learning workflows, including:

- Data preprocessing and feature engineering
- Model training and evaluation with standardized interfaces
- Model serialization and management
- Visualization utilities for learning curves and feature importance
- Integration with scikit-learn and other popular ML frameworks
- Extensive test suite with >95% code coverage
- Comprehensive documentation and practical examples
- Robust logging and hierarchical error handling system
- Type hints throughout the codebase (compatible with mypy)

## Installation

```bash
# Install the package
poetry add ml-library

# Alternative using pip
pip install ml-library
```

### Development Installation

Clone the repository and install with Poetry:

```bash
git clone https://github.com/longhoag/ml_library.git
cd ml_library
poetry install
```

To install with optional dependencies:

```bash
# Install with TensorFlow support
poetry install -E tensorflow

# Install with PyTorch support
poetry install -E torch

# Install with all extras
poetry install -E tensorflow -E torch
```

## Quick Start

```python
from ml_library import (
    LogisticModel,
    StandardPreprocessor,
    configure_logging,
    get_logger,
    accuracy
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Configure logging
configure_logging(level="info", log_file="ml_library.log")
logger = get_logger(__name__)

try:
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess the data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train model
    model = LogisticModel()
    model.train(X_train_processed, y_train)

    # Evaluate
    metrics = model.evaluate(X_test_processed, y_test)
    logger.info("Model accuracy: %.4f", metrics["accuracy"])

    # You can also use individual metrics
    acc = accuracy(y_test, model.predict(X_test_processed))
    logger.info("Accuracy calculated manually: %.4f", acc)

    # Save model
    model.save("model.pkl")
    logger.info("Model saved to model.pkl")

except Exception as e:
    logger.exception("Error in ML workflow: %s", str(e))
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

## Main Components

### Models

The library provides several model implementations:

- **LinearModel**: Linear regression model
- **LogisticModel**: Logistic regression for classification
- **RandomForestModel**: Random forest for classification
- **RandomForestRegressorModel**: Random forest for regression

All models share a consistent interface:
```python
model.train(X_train, y_train)  # Train the model
predictions = model.predict(X_test)  # Make predictions
metrics = model.evaluate(X_test, y_test)  # Evaluate performance
model.save("model.pkl")  # Serialize the model
loaded_model = ModelClass.load("model.pkl")  # Load a saved model
```

### Preprocessing

Data preprocessing components:

- **StandardPreprocessor**: Standardizes numerical features
- **PolynomialPreprocessor**: Generates polynomial features
- **FeatureSelector**: Selects the most important features

Example:
```python
preprocessor = StandardPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
```

### Metrics

The library includes functions for common evaluation metrics:

- Classification: `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- Regression: `mse`, `mae`, `r2`

### Visualization

Visualization utilities include:

- `plot_learning_curve`: Plot learning curves for model training
- `plot_feature_importances`: Visualize feature importance

## Testing

The library has a comprehensive test suite using pytest with >95% code coverage:

```bash
# Run all tests
python -m pytest

# Run tests with coverage report
python -m pytest --cov=src/ml_library --cov-report term
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
poetry run pytest --cov=ml_library

# Generate HTML coverage report
poetry run pytest --cov=ml_library --cov-report=html
```

For more details on testing, see [tests/README.md](tests/README.md).

## Documentation

Full documentation is available at [docs link].

## Versioning and Releases

This project follows [Semantic Versioning](https://semver.org/). For the versions available, see the [tags on this repository](https://github.com/longhoag/ml_library/tags).

See [CHANGELOG.md](CHANGELOG.md) for a list of all notable changes to the project.

## Distribution and Packaging

The library is distributed as a Python package via PyPI. For detailed instructions on building, versioning, and distributing the package, see [DISTRIBUTION.md](DISTRIBUTION.md).

To install the latest development version directly from the repository:

```bash
# Using Poetry
poetry add git+https://github.com/longhoag/ml_library.git

# Using pip
pip install git+https://github.com/longhoag/ml_library.git
```

## Contributing

Contributions are welcome! Please check our [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

We now use [Poetry](https://python-poetry.org/) for package management. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for development setup instructions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
