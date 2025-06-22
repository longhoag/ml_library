"""Tests for the models module."""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from ml_library.models import (
    LinearModel,
    LogisticModel,
    Model,
    RandomForestModel,
    RandomForestRegressorModel,
)


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


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        n_informative=3,
        random_state=42,
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


class TestBaseModel:
    """Test the base Model class."""

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        model = Model()
        assert model.trained is False
        assert model.model is None

    def test_model_train_not_implemented(self, classification_data):
        """Test that the train method raises NotImplementedError."""
        X_train, _, y_train, _ = classification_data
        model = Model()
        with pytest.raises(NotImplementedError):
            model.train(X_train, y_train)

    def test_model_predict_not_implemented(self, classification_data):
        """Test that the predict method raises NotImplementedError."""
        _, X_test, _, _ = classification_data
        model = Model()
        with pytest.raises(NotImplementedError):
            model.predict(X_test)

    def test_model_evaluate_not_implemented(self, classification_data):
        """Test that the evaluate method raises NotImplementedError."""
        _, X_test, _, y_test = classification_data
        model = Model()
        with pytest.raises(NotImplementedError):
            model.evaluate(X_test, y_test)


class TestLinearModel:
    """Test the LinearModel class."""

    def test_model_initialization(self):
        """Test that the LinearModel initializes correctly."""
        model = LinearModel()
        assert model.trained is False
        assert model.model is not None

    def test_model_train(self, regression_data):
        """Test training the model."""
        X_train, _, y_train, _ = regression_data
        model = LinearModel()
        trained_model = model.train(X_train, y_train)
        assert model.trained is True
        assert trained_model is model
        assert hasattr(model.model, "coef_")

    def test_model_predict(self, regression_data):
        """Test model prediction."""
        X_train, X_test, y_train, _ = regression_data
        model = LinearModel().train(X_train, y_train)
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_model_evaluate(self, regression_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = regression_data
        model = LinearModel().train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert 0 <= metrics["r2"] <= 1.0


class TestLogisticModel:
    """Test the LogisticModel class."""

    def test_model_initialization(self):
        """Test that the LogisticModel initializes correctly."""
        model = LogisticModel()
        assert model.trained is False
        assert model.model is not None

    def test_model_train(self, classification_data):
        """Test training the model."""
        X_train, _, y_train, _ = classification_data
        model = LogisticModel()
        trained_model = model.train(X_train, y_train)
        assert model.trained is True
        assert trained_model is model
        assert hasattr(model.model, "coef_")

    def test_model_predict(self, classification_data):
        """Test model prediction."""
        X_train, X_test, y_train, _ = classification_data
        model = LogisticModel().train(X_train, y_train)
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)
        # Check that predictions are either 0 or 1
        assert np.all(np.logical_or(predictions == 0, predictions == 1))

    def test_model_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, _ = classification_data
        model = LogisticModel().train(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(X_test)
        # Check that probabilities are between 0 and 1
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_model_evaluate(self, classification_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = classification_data
        model = LogisticModel().train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1.0


class TestRandomForestModel:
    """Test the RandomForestModel class."""

    def test_model_initialization(self):
        """Test that the RandomForestModel initializes correctly."""
        model = RandomForestModel()
        assert model.trained is False
        assert model.model is not None

    def test_model_train(self, classification_data):
        """Test training the model."""
        X_train, _, y_train, _ = classification_data
        model = RandomForestModel()
        trained_model = model.train(X_train, y_train)
        assert model.trained is True
        assert trained_model is model
        assert hasattr(model.model, "feature_importances_")

    def test_model_predict(self, classification_data):
        """Test model prediction."""
        X_train, X_test, y_train, _ = classification_data
        model = RandomForestModel().train(X_train, y_train)
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_model_predict_proba(self, classification_data):
        """Test probability prediction."""
        X_train, X_test, y_train, _ = classification_data
        model = RandomForestModel().train(X_train, y_train)
        probabilities = model.predict_proba(X_test)
        assert isinstance(probabilities, np.ndarray)
        assert len(probabilities) == len(X_test)
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_model_evaluate(self, classification_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = classification_data
        model = RandomForestModel().train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1.0


class TestRandomForestRegressorModel:
    """Test the RandomForestRegressorModel class."""

    def test_model_initialization(self):
        """Test that the RandomForestRegressorModel initializes correctly."""
        model = RandomForestRegressorModel()
        assert model.trained is False
        assert model.model is not None

    def test_model_train(self, regression_data):
        """Test training the model."""
        X_train, _, y_train, _ = regression_data
        model = RandomForestRegressorModel()
        trained_model = model.train(X_train, y_train)
        assert model.trained is True
        assert trained_model is model
        assert hasattr(model.model, "feature_importances_")

    def test_model_predict(self, regression_data):
        """Test model prediction."""
        X_train, X_test, y_train, _ = regression_data
        model = RandomForestRegressorModel().train(X_train, y_train)
        predictions = model.predict(X_test)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X_test)

    def test_model_evaluate(self, regression_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = regression_data
        model = RandomForestRegressorModel().train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert 0 <= metrics["r2"] <= 1.0
