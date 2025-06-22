"""Tests for model saving and loading."""

import os
import tempfile

import numpy as np
import pytest

from ml_library.exceptions import NotFittedError
from ml_library.models import LinearModel, LogisticModel, RandomForestModel


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample data for testing."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_reg = np.array([2, 4, 6, 8])  # Linear relationship for regression
    y_cls = np.array([0, 0, 1, 1])  # Binary classification
    return X, y_reg, y_cls


def test_save_untrained_model() -> None:
    """Test saving an untrained model raises error."""
    model = LinearModel()
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        with pytest.raises(NotFittedError):
            model.save(tmp.name)


def test_save_load_linear_model(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test saving and loading a LinearModel."""
    X, y, _ = sample_data

    # Train the model
    model = LinearModel()
    model.train(X, y)

    # Get predictions before saving
    original_preds = model.predict(X)

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    try:
        # Save and check return value
        saved_model = model.save(model_path)
        assert saved_model is model  # save should return self

        # Load the model
        loaded_model = LinearModel.load(model_path)

        # Check the loaded model has the same attributes
        assert isinstance(loaded_model, LinearModel)
        assert loaded_model.fitted is True
        assert loaded_model.fit_intercept == model.fit_intercept

        # Check predictions are the same
        loaded_preds = loaded_model.predict(X)
        assert np.allclose(original_preds, loaded_preds)

        # Check evaluation metrics
        original_metrics = model.evaluate(X, y)
        loaded_metrics = loaded_model.evaluate(X, y)

        for key in original_metrics:
            assert np.isclose(original_metrics[key], loaded_metrics[key])

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_save_load_logistic_model(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test saving and loading a LogisticModel."""
    X, _, y = sample_data

    # Train the model
    model = LogisticModel()
    model.train(X, y)

    # Get predictions before saving
    original_preds = model.predict(X)
    original_probs = model.predict_proba(X)

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    try:
        # Save model
        model.save(model_path)

        # Load the model
        loaded_model = LogisticModel.load(model_path)

        # Check the loaded model has the same attributes
        assert isinstance(loaded_model, LogisticModel)
        assert loaded_model.fitted is True
        assert loaded_model.C == model.C
        assert loaded_model.max_iter == model.max_iter

        # Check predictions are the same
        loaded_preds = loaded_model.predict(X)
        loaded_probs = loaded_model.predict_proba(X)

        assert np.array_equal(original_preds, loaded_preds)
        assert np.allclose(original_probs, loaded_probs)

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_save_load_random_forest_model(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test saving and loading a RandomForestModel."""
    X, _, y = sample_data

    # Train the model
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.train(X, y)

    # Get predictions and feature importances before saving
    original_preds = model.predict(X)
    original_importances = model.feature_importances_

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name

    try:
        # Save model
        model.save(model_path)

        # Load the model
        loaded_model = RandomForestModel.load(model_path)

        # Check the loaded model has the same attributes
        assert isinstance(loaded_model, RandomForestModel)
        assert loaded_model.fitted is True
        assert loaded_model.n_estimators == model.n_estimators
        assert loaded_model.random_state == model.random_state

        # Check predictions are the same
        loaded_preds = loaded_model.predict(X)
        loaded_importances = loaded_model.feature_importances_

        assert np.array_equal(original_preds, loaded_preds)
        assert np.array_equal(original_importances, loaded_importances)

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


def test_load_nonexistent_file() -> None:
    """Test loading a model from a nonexistent file raises IOError."""
    nonexistent_path = "/path/to/nonexistent/model.pkl"
    with pytest.raises(IOError):
        LinearModel.load(nonexistent_path)


def test_load_wrong_model_type(sample_data: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Test loading a model into the wrong class type raises ValueError."""
    X, y, _ = sample_data

    # Train a LinearModel
    model = LinearModel()
    model.train(X, y)

    # Save the model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        model_path = tmp.name
        model.save(model_path)

    try:
        # Try loading as LogisticModel (wrong type)
        with pytest.raises(ValueError):
            LogisticModel.load(model_path)

    finally:
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
