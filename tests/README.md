# Testing Framework

This document outlines the testing approach for the ML Library.

## Test Structure

Tests are organized in the following modules:

- `test_base.py`: Basic tests for the core model and preprocessor classes
- `test_metrics.py`: Tests for classification and regression metrics
- `test_models.py`: Tests for all model implementations
- `test_preprocessing.py`: Tests for data preprocessing classes
- `test_utils.py`: Tests for utility functions
- `test_visualization.py`: Tests for visualization functions

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run tests with coverage:

```bash
python -m pytest --cov=ml_library
```

For a detailed coverage report:

```bash
python -m pytest --cov=ml_library --cov-report=html
```

## Continuous Integration

The project uses GitHub Actions for continuous integration. The workflow is defined in `.github/workflows/test.yml` and includes:

1. Testing on multiple Python versions (3.8, 3.9, 3.10)
2. Linting with flake8
3. Type checking with mypy
4. Running tests with pytest
5. Reporting coverage to Codecov

## Test Coverage

The current test coverage is approximately 82%.

## Known Issues

There are currently some test failures related to:

1. Base model implementation returning wrong exception types
2. Model evaluation methods not returning dictionaries as expected
3. Missing attributes in preprocessing classes
4. Cross-validation utility needing interface adjustments

These issues will be addressed in future updates.
