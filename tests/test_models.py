"""Tests for model implementations."""

import os
from typing import Any, Dict, Generator, Optional, Tuple, Type

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_library.exceptions import NotFittedError
from ml_library.models import Model


# Define MockModel outside fixture to make it picklable
class MockModel(Model):
    """Mock model for testing."""

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Model:
        """Train the model."""
        self.fitted = True
        return self

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions."""
        if not self.fitted:
            raise NotFittedError("Model not trained")
        return np.zeros(len(X), dtype=np.float64)

    def evaluate(
        self,
        X: NDArray[Any],
        y: NDArray[Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.fitted:
            raise NotFittedError("Model not trained")
        return {"mock_metric": 1.0}


@pytest.fixture
def simple_dataset() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create a simple dataset for testing.

    Returns
    -------
    Tuple[NDArray, NDArray]
        X and y arrays for testing.
    """
    X = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([0, 1], dtype=np.float64)
    return X, y


@pytest.fixture
def model_class() -> Type[Model]:
    """Create a mock model class for testing.

    Returns
    -------
    Type[Model]
        Mock model class.
    """
    return MockModel


def test_model_init(model_class: Type[Model]) -> None:
    """Test model initialization."""
    model = model_class()
    assert not model.fitted


def test_model_train(
    model_class: Type[Model], simple_dataset: Tuple[NDArray[Any], NDArray[Any]]
) -> None:
    """Test model training."""
    model = model_class()
    X, y = simple_dataset
    result = model.train(X, y)
    assert result is model
    assert model.fitted


def test_model_predict_without_training(
    model_class: Type[Model], simple_dataset: Tuple[NDArray[Any], NDArray[Any]]
) -> None:
    """Test that prediction without training raises an error."""
    model = model_class()
    X, _ = simple_dataset
    with pytest.raises(NotFittedError):
        model.predict(X)


def test_model_predict_after_training(
    model_class: Type[Model], simple_dataset: Tuple[NDArray[Any], NDArray[Any]]
) -> None:
    """Test prediction after training."""
    model = model_class()
    X, y = simple_dataset
    model.train(X, y)
    predictions = model.predict(X)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)


def test_model_evaluate_without_training(
    model_class: Type[Model], simple_dataset: Tuple[NDArray[Any], NDArray[Any]]
) -> None:
    """Test that evaluation without training raises an error."""
    model = model_class()
    X, y = simple_dataset
    with pytest.raises(NotFittedError):
        model.evaluate(X, y)


def test_model_evaluate_after_training(
    model_class: Type[Model], simple_dataset: Tuple[NDArray[Any], NDArray[Any]]
) -> None:
    """Test evaluation after training."""
    model = model_class()
    X, y = simple_dataset
    model.train(X, y)
    metrics = model.evaluate(X, y)
    assert isinstance(metrics, dict)
    assert "mock_metric" in metrics
    assert metrics["mock_metric"] == 1.0


def test_model_save_and_load(
    model_class: Type[Model],
    simple_dataset: Tuple[NDArray[Any], NDArray[Any]],
    tmp_path: str,
) -> None:
    """Test model saving and loading."""
    model = model_class()
    X, y = simple_dataset
    model.train(X, y)

    # Save the model
    save_path = os.path.join(tmp_path, "model.joblib")
    model.save(save_path)
    assert os.path.exists(save_path)

    # Load the model
    loaded_model = model_class.load(save_path)
    assert isinstance(loaded_model, model_class)
    assert loaded_model.fitted

    # Test that the loaded model can make predictions
    predictions_original = model.predict(X)
    predictions_loaded = loaded_model.predict(X)
    np.testing.assert_array_equal(predictions_original, predictions_loaded)
