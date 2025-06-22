"""Tests for feature_engineering module."""

import numpy as np
import pytest

# DataError not used in this test file
from ml_library.preprocessing.feature_engineering import (
    FeatureSelector,
    PolynomialPreprocessor,
)


def test_feature_selector_properties():
    """Test the properties of FeatureSelector."""
    # Create dummy data
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([0, 1, 0, 1])

    # Test with automatic k selection
    selector = FeatureSelector(k=2)
    selector.fit(X, y)

    # Test get_support property
    support = selector.get_support
    assert isinstance(support, np.ndarray)
    assert support.dtype == bool

    # Test scores property
    scores = selector.scores_
    assert isinstance(scores, np.ndarray)
    assert len(scores) == 3  # One score per feature

    # Test with predefined feature indices
    selector_manual = FeatureSelector(feature_indices=[0, 2])
    selector_manual.fit(X)  # Need to call fit first, even if it's a no-op

    support_manual = selector_manual.get_support
    assert isinstance(support_manual, np.ndarray)
    assert support_manual.dtype == bool

    # Scores should be None for predefined indices
    assert selector_manual.scores_ is None


def test_polynomial_preprocessor_edge_cases():
    """Test edge cases in PolynomialPreprocessor."""
    # Test with interaction_only
    X = np.array([[1, 2], [3, 4]])

    poly_interact = PolynomialPreprocessor(degree=2, interaction_only=True)
    X_interact = poly_interact.fit_transform(X)

    # Should have features: [1(bias), x1, x2, x1*x2]
    assert X_interact.shape[1] == 4


if __name__ == "__main__":
    pytest.main()
