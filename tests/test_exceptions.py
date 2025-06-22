"""Tests for the exceptions module."""

import pytest

from ml_library.exceptions import (
    DataError,
    InvalidParameterError,
    MLLibraryError,
    ModelError,
    NotFittedError,
    PreprocessingError,
    ValidationError,
)


def test_ml_library_error() -> None:
    """Test MLLibraryError initialization."""
    # Test with default message
    error = MLLibraryError()
    assert str(error) == "An error occurred in ML Library"

    # Test with custom message
    error = MLLibraryError("Custom error message")
    assert str(error) == "Custom error message"
    assert error.message == "Custom error message"


def test_data_error() -> None:
    """Test DataError initialization."""
    # Test with default message
    error = DataError()
    assert str(error) == "Invalid data"
    assert error.data_shape is None

    # Test with custom message
    error = DataError("Data validation failed")
    assert str(error) == "Data validation failed"
    assert error.data_shape is None

    # Test with data shape
    error = DataError("Invalid dimensions", data_shape=(10, 5))
    assert str(error) == "Invalid dimensions (shape: (10, 5))"
    assert error.data_shape == (10, 5)


def test_model_error() -> None:
    """Test ModelError initialization."""
    # Test with default message
    error = ModelError()
    assert str(error) == "Model error"
    assert error.model_type is None
    assert error.details == {}

    # Test with custom message
    error = ModelError("Failed to train model")
    assert str(error) == "Failed to train model"

    # Test with model type
    error = ModelError("Training failed", model_type="RandomForest")
    assert str(error) == "Training failed in RandomForest"
    assert error.model_type == "RandomForest"

    # Test with details
    details = {"accuracy": 0.7, "error": "poor performance"}
    error = ModelError(
        "Evaluation failed", model_type="LogisticRegression", details=details
    )
    assert "Evaluation failed in LogisticRegression (details:" in str(error)
    assert error.details == details


def test_not_fitted_error() -> None:
    """Test NotFittedError initialization."""
    # Test with default message
    error = NotFittedError()
    assert str(error) == "Model is not fitted"
    assert error.model_type is None

    # Test with custom message and model type
    error = NotFittedError(
        "Cannot predict with untrained model", model_type="LinearRegression"
    )
    assert str(error) == "Cannot predict with untrained model in LinearRegression"
    assert error.model_type == "LinearRegression"


def test_invalid_parameter_error() -> None:
    """Test InvalidParameterError initialization."""
    # Test with minimal parameters
    error = InvalidParameterError("learning_rate", -0.1)
    assert str(error) == "Invalid value for parameter 'learning_rate': -0.1"
    assert error.param_name == "learning_rate"
    assert error.param_value == -0.1
    assert error.allowed_values is None

    # Test with allowed values
    error = InvalidParameterError(
        "activation", "tanh", allowed_values=["relu", "sigmoid"]
    )
    expected_msg = (
        "Invalid value for parameter 'activation': tanh, "
        "allowed values are: ['relu', 'sigmoid']"
    )
    assert str(error) == expected_msg
    assert error.allowed_values == ["relu", "sigmoid"]

    # Test with custom message
    error = InvalidParameterError(
        "n_estimators", -5, message="n_estimators must be positive"
    )
    assert str(error) == "n_estimators must be positive"


def test_preprocessing_error() -> None:
    """Test PreprocessingError initialization."""
    # Test with default message
    error = PreprocessingError()
    assert str(error) == "Preprocessing error"
    assert error.preprocessor_type is None
    assert error.data_shape is None

    # Test with custom message
    error = PreprocessingError("Failed to normalize data")
    assert str(error) == "Failed to normalize data"

    # Test with preprocessor type
    error = PreprocessingError("Scaling failed", preprocessor_type="StandardScaler")
    assert str(error) == "Scaling failed (preprocessor: StandardScaler)"
    assert error.preprocessor_type == "StandardScaler"

    # Test with data shape
    error = PreprocessingError("Missing values detected", data_shape=(100, 10))
    assert str(error) == "Missing values detected (data shape: (100, 10))"
    assert error.data_shape == (100, 10)

    # Test with both preprocessor type and data shape
    error = PreprocessingError(
        "Feature extraction failed",
        preprocessor_type="PolynomialFeatures",
        data_shape=(50, 5),
    )
    expected = (
        "Feature extraction failed (preprocessor: PolynomialFeatures, "
        "data shape: (50, 5))"
    )
    assert str(error) == expected
    assert error.preprocessor_type == "PolynomialFeatures"
    assert error.data_shape == (50, 5)


def test_validation_error() -> None:
    """Test ValidationError initialization."""
    # Test with default message
    error = ValidationError()
    assert str(error) == "Validation error"
    assert error.context == {}

    # Test with custom message
    error = ValidationError("Parameter validation failed")
    assert str(error) == "Parameter validation failed"

    # Test with context
    context = {"param": "n_clusters", "min_value": 2}
    error = ValidationError("Invalid clustering parameters", context=context)
    expected = (
        "Invalid clustering parameters (context: "
        "{'param': 'n_clusters', 'min_value': 2})"
    )
    assert str(error) == expected
    assert error.context == context


def test_inheritance() -> None:
    """Test inheritance hierarchy of exceptions."""
    # Test that all errors are instances of MLLibraryError
    assert issubclass(DataError, MLLibraryError)
    assert issubclass(ModelError, MLLibraryError)
    assert issubclass(NotFittedError, ModelError)  # NotFittedError is a ModelError
    assert issubclass(InvalidParameterError, MLLibraryError)
    assert issubclass(PreprocessingError, MLLibraryError)
    assert issubclass(ValidationError, MLLibraryError)

    # Test that all errors are also instances of built-in Exception
    assert issubclass(MLLibraryError, Exception)

    # Test exception catching
    try:
        raise NotFittedError("Model not trained")
    except ModelError:
        # Should be caught as ModelError
        pass
    else:
        pytest.fail("NotFittedError should be caught as ModelError")

    try:
        raise DataError("Invalid data")
    except MLLibraryError:
        # Should be caught as MLLibraryError
        pass
    else:
        pytest.fail("DataError should be caught as MLLibraryError")
