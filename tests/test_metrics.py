"""Test module for evaluation metrics."""
import unittest

import numpy as np

from ml_library.metrics import accuracy, f1, mae, mse, precision, r2, recall, roc_auc


class TestMetrics(unittest.TestCase):
    """Test case for evaluation metrics."""

    def setUp(self) -> None:
        """Initialize test data."""
        # Regression metrics test data
        self.y_true_reg = np.array([3, -0.5, 2, 7], dtype=np.float64)
        self.y_pred_reg = np.array([2.5, 0.0, 2, 8], dtype=np.float64)

        # Classification metrics test data
        self.y_true_clf = np.array([0, 1, 1, 0], dtype=np.float64)
        self.y_pred_clf = np.array([0, 1, 0, 0], dtype=np.float64)
        self.y_scores = np.array([0.1, 0.8, 0.3, 0.2], dtype=np.float64)

    def test_mse(self) -> None:
        """Verify mean squared error calculation."""
        error = mse(self.y_true_reg, self.y_pred_reg)
        self.assertAlmostEqual(error, 0.375)

    def test_mae(self) -> None:
        """Verify mean absolute error calculation."""
        error = mae(self.y_true_reg, self.y_pred_reg)
        self.assertAlmostEqual(error, 0.5)

    def test_r2(self) -> None:
        """Verify R² score calculation."""
        score = r2(self.y_true_reg, self.y_pred_reg)
        self.assertAlmostEqual(score, 0.949, places=3)

    def test_accuracy(self) -> None:
        """Verify accuracy score calculation."""
        score = accuracy(self.y_true_clf, self.y_pred_clf)
        self.assertAlmostEqual(score, 0.75)

    def test_precision(self) -> None:
        """Verify precision score calculation."""
        score = precision(self.y_true_clf, self.y_pred_clf)
        self.assertAlmostEqual(score, 1.0)

    def test_recall(self) -> None:
        """Verify recall score calculation."""
        score = recall(self.y_true_clf, self.y_pred_clf)
        self.assertAlmostEqual(score, 0.5)

    def test_f1(self) -> None:
        """Verify F1 score calculation."""
        score = f1(self.y_true_clf, self.y_pred_clf)
        self.assertAlmostEqual(score, 0.667, places=3)

    def test_roc_auc(self) -> None:
        """Verify ROC AUC score calculation."""
        score = roc_auc(self.y_true_clf, self.y_scores)
        self.assertAlmostEqual(score, 1.0)
