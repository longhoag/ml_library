"""Extended tests for utils module."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from ml_library.exceptions import DataError
from ml_library.utils import check_data, cross_validate


class SimpleMockModel(BaseEstimator):
    """Simple mock model for testing cross_validate."""

    def __init__(self) -> None:
        self.fitted = False
        self.train_count = 0
        self.eval_count = 0

    def fit(self, X, y):
        """Mock fit method."""
        self.fitted = True
        self.train_count += 1
        return self

    def predict(self, X):
        """Mock predict method."""
        return np.zeros(len(X))

    def score(self, X, y):
        """Mock score method."""
        self.eval_count += 1
        return 0.8  # Fixed score for testing


def test_check_data_corner_cases():
    """Test check_data with various edge cases."""
    # Test with 1D array
    X = np.array([1, 2, 3, 4])
    with pytest.raises(DataError):
        check_data(X, ensure_2d=True)

    # Test with 1D array, not enforcing 2D
    X_checked, _ = check_data(X, ensure_2d=False)
    assert len(X_checked) == 4

    # Test with empty array
    with pytest.raises(DataError):
        check_data(np.array([]))

    # Test with non-numeric data
    with pytest.raises(DataError):
        check_data(np.array([["a", "b"], ["c", "d"]]))

    # Test with NaN values
    X_with_nan = np.array([[1, 2], [3, np.nan]])
    with pytest.raises(DataError):
        check_data(X_with_nan)

    # Test with infinity values
    X_with_inf = np.array([[1, 2], [3, np.inf]])
    with pytest.raises(DataError):
        check_data(X_with_inf)


def test_cross_validate_edge_cases():
    """Test cross_validate with edge cases."""
    # Test with small dataset
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model = SimpleMockModel()

    # Should run with warning for small dataset
    results = cross_validate(model, X, y, cv=2)
    assert "test_score" in results
    assert "train_score" in results
    assert len(results["test_score"]) == 2
    assert model.train_count >= 2

    # Test with custom scoring metric
    results = cross_validate(model, X, y, cv=2, scoring="precision")
    assert "test_score" in results

    # Test with 1D array for X (should raise error)
    X_1d = np.array([1, 2, 3, 4])
    y_1d = np.array([0, 1, 0, 1])
    with pytest.raises(Exception):
        cross_validate(model, X_1d, y_1d, cv=2)

    # Test with mismatched X, y lengths
    X_mismatch = np.array([[1, 2], [3, 4], [5, 6]])
    y_mismatch = np.array([0, 1])
    with pytest.raises(Exception):
        cross_validate(model, X_mismatch, y_mismatch, cv=2)
