"""Extended tests for preprocessing module."""

import numpy as np
import pytest

from ml_library.exceptions import PreprocessingError
from ml_library.preprocessing import (
    FeatureSelector,
    PolynomialPreprocessor,
    StandardPreprocessor,
)


class TestStandardPreprocessorExtended:
    """Extended tests for StandardPreprocessor."""

    def test_transform_without_fit(self) -> None:
        """Test that transform raises error if not fitted."""
        preprocessor = StandardPreprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(PreprocessingError):
            preprocessor.transform(X)

    def test_transform_wrong_shape(self) -> None:
        """Test transform with wrong shape input."""
        preprocessor = StandardPreprocessor()
        X_train = np.array([[1, 2], [3, 4]], dtype=np.float64)
        preprocessor.fit(X_train)

        # Test with wrong number of features
        X_test = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        with pytest.raises(PreprocessingError):
            preprocessor.transform(X_test)

    def test_fit_transform_edge_cases(self) -> None:
        """Test fit_transform with edge cases."""
        preprocessor = StandardPreprocessor()

        # Test with constant features
        X = np.array([[1, 5], [1, 5], [1, 5]], dtype=np.float64)
        X_transformed = preprocessor.fit_transform(X)

        # First feature should be all zeros (constant gets normalized to 0)
        assert np.allclose(X_transformed[:, 0], 0)

        # Test memory of transformed values
        assert hasattr(preprocessor, "scaler")
        assert hasattr(preprocessor.scaler, "mean_")
        assert hasattr(preprocessor.scaler, "scale_")


class TestPolynomialPreprocessorExtended:
    """Extended tests for PolynomialPreprocessor."""

    def test_transform_without_fit(self) -> None:
        """Test that transform raises error if not fitted."""
        preprocessor = PolynomialPreprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(PreprocessingError):
            preprocessor.transform(X)

    def test_with_higher_degrees(self) -> None:
        """Test polynomial features with higher degrees."""
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        # Test with degree 3
        preprocessor = PolynomialPreprocessor(degree=3)
        X_poly = preprocessor.fit_transform(X)

        # With degree 3 and include_bias=True, we should have:
        # 1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
        # So 10 features for each sample
        assert X_poly.shape == (2, 10)

        # First row should be: [1, 1, 2, 1, 2, 4, 1, 2, 4, 8]
        expected_first_row = np.array([1, 1, 2, 1, 2, 4, 1, 2, 4, 8], dtype=np.float64)
        assert np.allclose(X_poly[0], expected_first_row)


class TestFeatureSelectorExtended:
    """Extended tests for FeatureSelector."""

    def test_k_validation(self) -> None:
        """Test with invalid k values."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        # Test with k=0
        with pytest.raises(ValueError):
            FeatureSelector(k=0)

        # Test with k > n_features
        with pytest.raises(ValueError):
            selector = FeatureSelector(k=4)
            selector.fit(X)

    def test_transform_without_fit(self) -> None:
        """Test that transform requires explicit fit."""
        selector = FeatureSelector(k=2)
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

        with pytest.raises(PreprocessingError):
            selector.transform(X)

    def test_k_features_selected(self) -> None:
        """Test that k features are selected."""
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        k = 2
        selector = FeatureSelector(k=k)
        X_transformed = selector.fit_transform(X)
        assert X_transformed.shape == (2, k)


def test_combining_preprocessors() -> None:
    """Test chaining preprocessors together."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)

    # Create a preprocessing pipeline:
    # 1. Select first feature
    # 2. Add polynomial features
    # 3. Standardize

    selector = FeatureSelector(k=1)
    X_selected = selector.fit_transform(X)
    assert X_selected.shape == (4, 1)

    poly = PolynomialPreprocessor(degree=2)
    X_poly = poly.fit_transform(X_selected)
    assert X_poly.shape == (4, 3)  # original + squared + intercept

    scaler = StandardPreprocessor()
    X_scaled = scaler.fit_transform(X_poly)
    assert X_scaled.shape == (4, 3)

    # Check the data is properly normalized
    assert np.isclose(X_scaled.mean(axis=0)[1], 0)  # Mean of each column should be ~0
    assert np.isclose(X_scaled.std(axis=0)[1], 1)  # Std of each column should be ~1
