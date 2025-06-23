"""Tests for ensemble models."""

from typing import Tuple

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from ml_library.models import RandomForestModel, RandomForestRegressorModel

# Type alias for test data to reduce line length
ClassificationTestData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
RegressionTestData = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture
def classification_data() -> ClassificationTestData:
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, random_state=42
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data() -> RegressionTestData:
    """Generate sample regression data."""
    # make_regression can return coef as a third value which we don't need
    X, y, _ = make_regression(
        n_samples=100, n_features=5, n_informative=3, random_state=42, coef=True
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


class TestRandomForestErrors:
    """Test error handling in RandomForestModel."""

    def test_predict_not_trained(self) -> None:
        """Test predict raises error when model is not trained."""
        model = RandomForestModel()
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2], [3, 4]]))

    def test_predict_proba_not_trained(self) -> None:
        """Test predict_proba raises error when model is not trained."""
        model = RandomForestModel()
        with pytest.raises(ValueError):
            model.predict_proba(np.array([[1, 2], [3, 4]]))

    def test_evaluate_not_trained(self) -> None:
        """Test evaluate raises error when model is not trained."""
        model = RandomForestModel()
        with pytest.raises(ValueError):
            model.evaluate(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_feature_importances_not_trained(self) -> None:
        """Test feature_importances_ raises error when model is not trained."""
        model = RandomForestModel()
        with pytest.raises(ValueError):
            _ = model.feature_importances_

    def test_classes_not_trained(self) -> None:
        """Test classes_ raises error when model is not trained."""
        model = RandomForestModel()
        with pytest.raises(ValueError):
            _ = model.classes_


class TestRandomForestRegressorErrors:
    """Test error handling in RandomForestRegressorModel."""

    def test_predict_not_trained(self) -> None:
        """Test predict raises error when model is not trained."""
        model = RandomForestRegressorModel()
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2], [3, 4]]))

    def test_evaluate_not_trained(self) -> None:
        """Test evaluate raises error when model is not trained."""
        model = RandomForestRegressorModel()
        with pytest.raises(ValueError):
            model.evaluate(np.array([[1, 2], [3, 4]]), np.array([1.0, 2.0]))

    def test_feature_importances_not_trained(self) -> None:
        """Test feature_importances_ raises error when model is not trained."""
        model = RandomForestRegressorModel()
        with pytest.raises(ValueError):
            _ = model.feature_importances_


def test_random_forest_multiclass(classification_data: ClassificationTestData) -> None:
    """Test RandomForestModel with multiclass data."""
    # Rename local variable to avoid name shadowing with the fixture
    test_data = classification_data
    X_train, X_test, y_train, _ = test_data

    # Convert to multiclass (0, 1, 2)
    y_train_multi = y_train.copy()
    y_train_multi[0:20] = 2  # Add a third class

    # Train multiclass model
    model = RandomForestModel(n_estimators=10, random_state=42)
    model.train(X_train, y_train_multi)

    # Check predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert len(np.unique(y_pred)) <= 3  # Maximum of 3 classes

    # Check probabilities
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape[0] == X_test.shape[0]  # Same number of samples
    assert np.allclose(np.sum(y_proba, axis=1), 1.0)  # Probabilities sum to 1

    # Test multiclass evaluation
    y_test_multi = np.zeros(X_test.shape[0])
    y_test_multi[0:5] = 1
    y_test_multi[5:10] = 2

    metrics = model.evaluate(X_test, y_test_multi)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert all(
        0 <= metrics[m] <= 1.0 for m in ["accuracy", "precision", "recall", "f1"]
    )


def test_random_forest_parameter_variations() -> None:
    """Test RandomForestModel with different parameter values."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])

    # Default parameters
    model1 = RandomForestModel()
    model1.train(X, y)

    # Custom n_estimators
    model2 = RandomForestModel(n_estimators=5)
    model2.train(X, y)
    assert model2.n_estimators == 5

    # Custom max_depth
    model3 = RandomForestModel(max_depth=2)
    model3.train(X, y)
    assert model3.max_depth == 2

    # Custom random_state
    model4 = RandomForestModel(random_state=100)
    model4.train(X, y)
    assert model4.random_state == 100

    # All custom parameters
    model5 = RandomForestModel(n_estimators=10, max_depth=3, random_state=42)
    model5.train(X, y)
    assert model5.n_estimators == 10
    assert model5.max_depth == 3
    assert model5.random_state == 42


def test_random_forest_regressor_parameter_variations() -> None:
    """Test RandomForestRegressorModel with different parameter values."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([2.5, 4.5, 6.5, 8.5, 10.5])

    # Default parameters
    model1 = RandomForestRegressorModel()
    model1.train(X, y)

    # Custom n_estimators
    model2 = RandomForestRegressorModel(n_estimators=5)
    model2.train(X, y)
    assert model2.n_estimators == 5

    # Custom max_depth
    model3 = RandomForestRegressorModel(max_depth=2)
    model3.train(X, y)
    assert model3.max_depth == 2

    # Custom random_state
    model4 = RandomForestRegressorModel(random_state=100)
    model4.train(X, y)
    assert model4.random_state == 100
