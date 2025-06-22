"""Additional tests for linear models to improve coverage."""

from typing import Any, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.datasets import make_classification, make_regression

from ml_library.exceptions import DataError, NotFittedError
from ml_library.models.linear import LinearModel, LogisticModel


@pytest.fixture
def classification_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample regression data."""
    X, y, _ = make_regression(
        n_samples=100, n_features=5, n_informative=3, random_state=42, coef=True
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


class TestLinearModelProperties:
    """Tests for LinearModel properties and error handling."""

    def test_coef_not_fitted(self) -> None:
        """Test coefficients property when model is not fitted."""
        model = LinearModel()
        with pytest.raises(NotFittedError):
            _ = model.coef_

    def test_intercept_not_fitted(self) -> None:
        """Test intercept property when model is not fitted."""
        model = LinearModel()
        with pytest.raises(NotFittedError):
            _ = model.intercept_

    def test_evaluate_error_handling(
        self, regression_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test error handling in evaluate method."""
        X_train, X_test, y_train, y_test = regression_data
        model = LinearModel()
        model.train(X_train, y_train)

        # Create a test scenario where evaluation would fail
        original_predict = model.predict

        # Use monkeypatch approach using a bound method
        from types import MethodType

        def mock_predict(
            self: Any, X: np.ndarray, **kwargs: Any
        ) -> NDArray[np.float64]:
            raise ValueError("Test error")

        try:
            # Replace predict with a method that raises an error
            # Store in __dict__ to avoid mypy complaints about method assignment
            model.__dict__["predict"] = MethodType(mock_predict, model)

            with pytest.raises(DataError):
                model.evaluate(X_test, y_test)
        finally:
            # Restore original function
            model.__dict__["predict"] = original_predict

    def test_custom_evaluation_metrics(
        self, regression_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test custom metrics in evaluate method."""
        X_train, X_test, y_train, y_test = regression_data
        model = LinearModel()
        model.train(X_train, y_train)

        def custom_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.sum(np.abs(y_true - y_pred)))

        metrics = model.evaluate(X_test, y_test, metrics={"custom": custom_metric})
        assert "custom" in metrics
        assert isinstance(metrics["custom"], float)


class TestLogisticModelProperties:
    """Tests for LogisticModel properties and error handling."""

    def test_coef_not_fitted(self) -> None:
        """Test coefficients property when model is not fitted."""
        model = LogisticModel()
        with pytest.raises(NotFittedError):
            _ = model.coef_

    def test_intercept_not_fitted(self) -> None:
        """Test intercept property when model is not fitted."""
        model = LogisticModel()
        with pytest.raises(NotFittedError):
            _ = model.intercept_

    def test_evaluate_error_handling(
        self, classification_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test error handling in evaluate method."""
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticModel()
        model.train(X_train, y_train)

        # Create a test scenario where evaluation would fail
        original_predict = model.predict

        # Use monkeypatch approach using a bound method
        from types import MethodType

        def mock_predict(
            self: Any, X: np.ndarray, **kwargs: Any
        ) -> NDArray[np.float64]:
            raise ValueError("Test error")

        try:
            # Replace predict with a method that raises an error
            model.__dict__["predict"] = MethodType(mock_predict, model)

            with pytest.raises(DataError):
                model.evaluate(X_test, y_test)
        finally:
            # Restore original function
            model.__dict__["predict"] = original_predict

    def test_custom_evaluation_metrics(
        self, classification_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        """Test custom metrics in evaluate method."""
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticModel()
        model.train(X_train, y_train)

        def custom_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean(y_true == y_pred))

        metrics = model.evaluate(
            X_test, y_test, metrics={"custom_acc": custom_accuracy}
        )
        assert "custom_acc" in metrics
        assert isinstance(metrics["custom_acc"], float)

    def test_multi_class_handling(self) -> None:
        """Test LogisticModel with multi-class data."""
        # Generate multi-class data
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=3,
            random_state=42,
        )
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        model = LogisticModel()
        model.train(X_train, y_train)

        # Test prediction
        y_pred = model.predict(X_test)
        assert y_pred.shape == y_test.shape

        # Test evaluation
        metrics = model.evaluate(X_test, y_test)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
