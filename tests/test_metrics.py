"""Tests for the metrics module."""
import numpy as np
import pytest

from ml_library.metrics import accuracy, precision, recall, f1, roc_auc, mse, mae, r2


class TestClassificationMetrics:
    """Test classification metrics."""

    @pytest.fixture
    def classification_data(self):
        """Example classification data for testing metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1])
        return y_true, y_pred

    @pytest.fixture
    def classification_proba_data(self):
        """Example classification probability data for testing metrics."""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.7, 0.8, 0.3, 0.6])
        return y_true, y_proba

    def test_accuracy(self, classification_data):
        """Test accuracy calculation."""
        y_true, y_pred = classification_data
        result = accuracy(y_true, y_pred)
        # 6 out of 8 correct predictions
        expected = 0.75
        assert result == pytest.approx(expected)

    def test_precision(self, classification_data):
        """Test precision calculation."""
        y_true, y_pred = classification_data
        result = precision(y_true, y_pred)
        # 3 true positives, 1 false positive
        expected = 0.75
        assert result == pytest.approx(expected)

    def test_recall(self, classification_data):
        """Test recall calculation."""
        y_true, y_pred = classification_data
        result = recall(y_true, y_pred)
        # 3 true positives, 1 false negative
        expected = 0.75
        assert result == pytest.approx(expected)

    def test_f1(self, classification_data):
        """Test F1 score calculation."""
        y_true, y_pred = classification_data
        result = f1(y_true, y_pred)
        # F1 = 2 * (precision * recall) / (precision + recall)
        # F1 = 2 * (0.75 * 0.75) / (0.75 + 0.75) = 0.75
        expected = 0.75
        assert result == pytest.approx(expected)

    def test_roc_auc(self, classification_proba_data):
        """Test ROC AUC calculation."""
        y_true, y_proba = classification_proba_data
        result = roc_auc(y_true, y_proba)
        # This is an approximate value for the test data
        assert 0.5 <= result <= 1.0


class TestRegressionMetrics:
    """Test regression metrics."""

    @pytest.fixture
    def regression_data(self):
        """Example regression data for testing metrics."""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        return y_true, y_pred

    def test_mse(self, regression_data):
        """Test mean squared error calculation."""
        y_true, y_pred = regression_data
        result = mse(y_true, y_pred)
        # MSE = ((3.0-2.5)² + (-0.5-0.0)² + (2.0-2.0)² + (7.0-8.0)²) / 4
        # MSE = (0.25 + 0.25 + 0.0 + 1.0) / 4 = 0.375
        expected = 0.375
        assert result == pytest.approx(expected)

    def test_mae(self, regression_data):
        """Test mean absolute error calculation."""
        y_true, y_pred = regression_data
        result = mae(y_true, y_pred)
        # MAE = (|3.0-2.5| + |-0.5-0.0| + |2.0-2.0| + |7.0-8.0|) / 4
        # MAE = (0.5 + 0.5 + 0.0 + 1.0) / 4 = 0.5
        expected = 0.5
        assert result == pytest.approx(expected)

    def test_r2(self, regression_data):
        """Test R² score calculation."""
        y_true, y_pred = regression_data
        result = r2(y_true, y_pred)
        # This test ensures r2 is between 0 and 1 for sensible predictions
        assert 0 <= result <= 1.0
