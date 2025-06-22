"""Extended tests for linear models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import NotFittedError as _SklearnNotFittedError  # noqa: F401

from ml_library.exceptions import DataError, NotFittedError
from ml_library.models import LinearModel, LogisticModel


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    # make_regression can return coef as a third value which we don't need
    X, y, _ = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42,
        coef=True,
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


class TestLinearModelErrors:
    """Test error handling in LinearModel."""

    def test_train_data_error(self):
        """Test that train raises DataError for invalid input."""
        model = LinearModel()
        with pytest.raises(DataError):
            # Invalid data (different dimensions)
            model.train(np.array([[1, 2], [3, 4]]), np.array([1, 2, 3]))

    def test_predict_not_fitted(self):
        """Test that predict raises NotFittedError if model not trained."""
        model = LinearModel()
        with pytest.raises(NotFittedError):
            model.predict(np.array([[1, 2], [3, 4]]))

    def test_predict_data_error(self):
        """Test that predict raises DataError for invalid input."""
        model = LinearModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        model.train(X_train, y_train)

        with pytest.raises(DataError):
            # Wrong number of features
            model.predict(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_evaluate_not_fitted(self):
        """Test that evaluate raises NotFittedError if model not trained."""
        model = LinearModel()
        with pytest.raises(NotFittedError):
            model.evaluate(np.array([[1, 2], [3, 4]]), np.array([1, 2]))

    def test_evaluate_data_error(self):
        """Test that evaluate raises DataError for invalid input."""
        model = LinearModel()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([1, 2, 3])
        model.train(X_train, y_train)

        with pytest.raises(DataError):
            # Wrong number of features
            model.evaluate(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2]))

    def test_property_access_not_fitted(self):
        """Test that accessing properties raises NotFittedError if model not trained."""
        model = LinearModel()

        with pytest.raises(NotFittedError):
            _ = model.coef_

        with pytest.raises(NotFittedError):
            _ = model.intercept_


class TestLogisticModelErrors:
    """Test error handling in LogisticModel."""

    def test_predict_data_error(self):
        """Test that predict handles invalid input."""
        model = LogisticModel()

        # Train the model
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([0, 1, 0])
        model.train(X_train, y_train)

        # Test with different number of features
        with pytest.raises(DataError):
            # Wrong number of features should raise an error
            model.predict(np.array([[1, 2, 3], [4, 5, 6]]))

    def test_predict_not_fitted(self):
        """Test that predict raises error if model not trained."""
        model = LogisticModel()

        with pytest.raises(NotFittedError):
            model.predict(np.array([[1, 2], [3, 4]]))

    def test_predict_proba_not_fitted(self):
        """Test that predict_proba raises error if model not trained."""
        model = LogisticModel()

        with pytest.raises(NotFittedError):
            model.predict_proba(np.array([[1, 2], [3, 4]]))

    def test_evaluate_not_fitted(self):
        """Test that evaluate raises error if model not trained."""
        model = LogisticModel()

        with pytest.raises(NotFittedError):
            model.evaluate(np.array([[1, 2], [3, 4]]), np.array([0, 1]))

    def test_property_access_not_fitted(self):
        """Test that accessing properties raises NotFittedError if model not trained."""
        model = LogisticModel()

        with pytest.raises(NotFittedError):
            _ = model.coef_

        with pytest.raises(NotFittedError):
            _ = model.intercept_

        with pytest.raises(NotFittedError):
            _ = model.classes_


def test_logistic_model_multiclass(classification_data):
    """Test LogisticModel with multiclass data."""
    # Rename local variable to avoid name shadowing with the fixture
    test_data = classification_data
    X_train, X_test, y_train, _ = test_data

    # Convert to multiclass (0, 1, 2)
    y_train_multi = y_train.copy()
    y_train_multi[0:20] = 2  # Add a third class

    # Train multiclass model
    model = LogisticModel()
    model.train(X_train, y_train_multi)

    # Check predictions
    y_pred = model.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)
    assert set(np.unique(y_pred)).issubset({0, 1, 2})

    # Check probabilities
    y_proba = model.predict_proba(X_test)
    assert y_proba.shape == (X_test.shape[0], 3)  # 3 classes
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
