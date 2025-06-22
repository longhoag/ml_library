"""Tests for the visualization module."""
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, ClassifierMixin

from ml_library.visualization import plot_learning_curve


class MockEstimator(BaseEstimator, ClassifierMixin):
    """Mock scikit-learn estimator for testing visualization."""

    def __init__(self):
        self.fitted = False

    def fit(self, X, y):
        self.fitted = True
        return self

    def predict(self, X):
        return np.ones(len(X))

    def score(self, X, y):
        return 0.9


def test_plot_learning_curve():
    """Test plot_learning_curve generates a figure."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model = MockEstimator()

    fig = plot_learning_curve(model, X, y, cv=3)

    assert isinstance(fig, Figure)


def test_plot_learning_curve_with_cross_validation():
    """Test plot_learning_curve with different cv values."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model = MockEstimator()

    fig = plot_learning_curve(model, X, y, cv=2)

    assert isinstance(fig, Figure)
