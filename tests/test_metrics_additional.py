"""Additional tests for metrics to improve coverage."""
import unittest

import numpy as np
import pytest

from ml_library.metrics import (
    accuracy, f1, mae, mse, precision, r2, recall, roc_auc
)


class TestMetricsEdgeCases(unittest.TestCase):
    """Test edge cases for metrics functions."""

    def test_precision_zero_division(self):
        """Test precision when there are no predicted positives."""
        y_true = np.array([0, 0, 0, 0], dtype=np.float64)
        y_pred = np.array([0, 0, 0, 0], dtype=np.float64)
        
        # Default behavior
        result = precision(y_true, y_pred)
        self.assertEqual(result, 0.0)
        
        # Custom zero_division value
        result = precision(y_true, y_pred, zero_division=0.5)
        self.assertEqual(result, 0.5)

    def test_recall_zero_division(self):
        """Test recall when there are no actual positives."""
        y_true = np.array([0, 0, 0, 0], dtype=np.float64)
        y_pred = np.array([1, 0, 1, 0], dtype=np.float64)
        
        # Default behavior
        result = recall(y_true, y_pred)
        self.assertEqual(result, 0.0)
        
        # Custom zero_division value
        result = recall(y_true, y_pred, zero_division=0.5)
        self.assertEqual(result, 0.5)

    def test_f1_zero_division(self):
        """Test F1 score when both precision and recall are 0."""
        # Case 1: No predicted positives
        y_true = np.array([1, 1, 1, 1], dtype=np.float64)
        y_pred = np.array([0, 0, 0, 0], dtype=np.float64)
        
        result = f1(y_true, y_pred)
        self.assertEqual(result, 0.0)
        
        # Case 2: No true positives, but both predicted and actual positives exist
        y_true = np.array([1, 1, 0, 0], dtype=np.float64)
        y_pred = np.array([0, 0, 1, 1], dtype=np.float64)
        
        result = f1(y_true, y_pred)
        self.assertEqual(result, 0.0)
        
        # Case 3: Custom zero_division value
        result = f1(y_true, y_pred, zero_division=0.5)
        self.assertEqual(result, 0.5)

    def test_r2_all_same_values(self):
        """Test R² score when all true values are the same."""
        y_true = np.array([5, 5, 5, 5], dtype=np.float64)
        y_pred = np.array([4, 6, 5, 7], dtype=np.float64)
        
        result = r2(y_true, y_pred)
        self.assertEqual(result, 0.0)

    def test_roc_auc_edge_cases(self):
        """Test ROC AUC score with edge cases."""
        # Case 1: No positive samples
        y_true = np.array([0, 0, 0, 0], dtype=np.float64)
        y_scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        
        result = roc_auc(y_true, y_scores)
        self.assertEqual(result, 0.5)
        
        # Case 2: No negative samples
        y_true = np.array([1, 1, 1, 1], dtype=np.float64)
        y_scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        
        result = roc_auc(y_true, y_scores)
        self.assertEqual(result, 0.5)
        
        # Case 3: Perfect separation
        y_true = np.array([0, 0, 1, 1], dtype=np.float64)
        y_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        
        result = roc_auc(y_true, y_scores)
        self.assertAlmostEqual(result, 1.0)


@pytest.mark.parametrize(
    "y_true, y_pred, expected",
    [
        # All correct
        (np.array([0, 0, 1, 1], dtype=np.float64), 
         np.array([0, 0, 1, 1], dtype=np.float64), 
         1.0),
        # All wrong
        (np.array([0, 0, 1, 1], dtype=np.float64), 
         np.array([1, 1, 0, 0], dtype=np.float64), 
         0.0),
        # Half correct
        (np.array([0, 0, 1, 1], dtype=np.float64), 
         np.array([0, 1, 1, 0], dtype=np.float64), 
         0.5),
    ]
)
def test_accuracy_parametrized(y_true, y_pred, expected):
    """Test accuracy with various scenarios using parametrization."""
    result = accuracy(y_true, y_pred)
    assert result == expected
