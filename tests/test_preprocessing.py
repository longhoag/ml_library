"""Tests for the preprocessing module."""

import numpy as np
import pytest

from ml_library.exceptions import PreprocessingError
from ml_library.preprocessing import (
    FeatureSelector,
    MinMaxScaler,
    PolynomialPreprocessor,
    Preprocessor,
    StandardPreprocessor,
    StandardScaler,
)


class TestBasePreprocessor:
    """Test the base Preprocessor class."""

    def test_preprocessor_init(self) -> None:
        """Test that the preprocessor initializes correctly."""
        preprocessor = Preprocessor()
        assert not preprocessor.fitted

    def test_preprocessor_fit(self) -> None:
        """Test that the preprocessor fit method works."""
        preprocessor = Preprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        result = preprocessor.fit(X, y)

        assert preprocessor.fitted
        assert result is preprocessor

    def test_preprocessor_transform_not_implemented(self) -> None:
        """Test that transform method raises NotImplementedError."""
        preprocessor = Preprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        preprocessor.fit(X, np.array([0, 1], dtype=np.float64))

        with pytest.raises(NotImplementedError):
            preprocessor.transform(X)

    def test_preprocessor_fit_transform(self) -> None:
        """Test that fit_transform calls fit and transform."""

        class MockPreprocessor(Preprocessor):
            def transform(self, X: np.ndarray) -> np.ndarray:
                return np.array([[10, 20], [30, 40]], dtype=np.float64)

        preprocessor = MockPreprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        result = preprocessor.fit_transform(X, y)

        assert preprocessor.fitted
        assert np.array_equal(result, np.array([[10, 20], [30, 40]], dtype=np.float64))


class TestStandardPreprocessor:
    """Test the StandardPreprocessor class."""

    def test_preprocessor_init(self) -> None:
        """Test that the StandardPreprocessor initializes correctly."""
        preprocessor = StandardPreprocessor()
        assert not preprocessor.fitted
        assert preprocessor.scaler is not None

    def test_preprocessor_fit(self) -> None:
        """Test that the StandardPreprocessor fit method works."""
        preprocessor = StandardPreprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        result = preprocessor.fit(X, y)

        assert preprocessor.fitted
        assert result is preprocessor

    def test_preprocessor_transform(self) -> None:
        """Test that transform method standardizes data."""
        preprocessor = StandardPreprocessor()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        # Check that the result has zero mean and unit variance (approximately)
        assert np.abs(result.mean()) < 1e-10
        assert np.abs(result.std() - 1.0) < 1e-10

    def test_preprocessor_with_feature_lists(self) -> None:
        """Test StandardPreprocessor with numerical and categorical features."""
        numerical_features = [0, 1]
        categorical_features = [2]
        preprocessor = StandardPreprocessor(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )

        # Test initialization
        assert preprocessor.numerical_features == numerical_features
        assert preprocessor.categorical_features == categorical_features
        assert preprocessor.scaler is not None
        assert preprocessor.transformer is not None

        # Create a dataset with mixed data types
        X = np.array(
            [[1.0, 2.0, 0], [3.0, 4.0, 1], [5.0, 6.0, 0], [7.0, 8.0, 1]], dtype=float
        )
        y = np.array([0, 1, 0, 1])

        # Test fitting
        preprocessor.fit(X, y)
        assert preprocessor.fitted

        # Test transformation
        transformed = preprocessor.transform(X)
        # Should have one-hot encoded categorical feature (2 columns) plus 2 numerical features
        assert transformed.shape[1] >= len(numerical_features) + 2


class TestPolynomialPreprocessor:
    """Test the PolynomialPreprocessor class."""

    def test_preprocessor_init(self) -> None:
        """Test that the PolynomialPreprocessor initializes correctly."""
        preprocessor = PolynomialPreprocessor(degree=2)
        assert not preprocessor.fitted
        assert preprocessor.transformer is not None
        assert preprocessor.degree == 2

    def test_preprocessor_fit(self) -> None:
        """Test that the PolynomialPreprocessor fit method works."""
        preprocessor = PolynomialPreprocessor(degree=2)
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        result = preprocessor.fit(X, y)

        assert preprocessor.fitted
        assert result is preprocessor

    def test_preprocessor_transform(self) -> None:
        """Test that transform method adds polynomial features."""
        preprocessor = PolynomialPreprocessor(
            degree=2, include_bias=True
        )  # include_bias=True to get the intercept term
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y = np.array([0, 1], dtype=np.float64)

        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        # For degree=2 and 2 features, with include_bias=True we should get 6 features:
        # [1, x1, x2, x1^2, x1*x2, x2^2]
        assert result.shape == (2, 6)


class TestFeatureSelector:
    """Test the FeatureSelector class."""

    def test_preprocessor_init(self) -> None:
        """Test that the FeatureSelector initializes correctly."""
        preprocessor = FeatureSelector(k=1)
        assert not preprocessor.fitted
        assert preprocessor.k == 1
        assert preprocessor.feature_indices is None

    def test_preprocessor_fit(self) -> None:
        """Test that the FeatureSelector fit method works."""
        preprocessor = FeatureSelector(k=1)
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
        y = np.array([0, 1, 0], dtype=np.float64)

        result = preprocessor.fit(X, y)

        assert preprocessor.fitted
        assert result is preprocessor
        assert preprocessor.feature_indices is not None
        assert len(preprocessor.feature_indices) == 1

    def test_preprocessor_transform(self) -> None:
        """Test that transform method selects top k features."""
        preprocessor = FeatureSelector(k=1)
        X = np.array([[1, 20], [3, 40], [5, 60]], dtype=np.float64)
        y = np.array([0, 1, 0], dtype=np.float64)

        preprocessor.fit(X, y)
        result = preprocessor.transform(X)

        # Should select the second column (index 1) because it has higher variance
        assert result.shape == (3, 1)
        assert np.array_equal(result, np.array([[20], [40], [60]], dtype=np.float64))


class TestStandardScaler:
    """Test the StandardScaler class."""

    def test_scaler_init(self) -> None:
        """Test that the StandardScaler initializes correctly."""
        scaler = StandardScaler()
        assert not scaler.fitted
        assert scaler.mean_ is None
        assert scaler.scale_ is None

    def test_scaler_fit(self) -> None:
        """Test that the StandardScaler fit method works."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        result = scaler.fit(X)

        assert scaler.fitted
        assert result is scaler
        assert np.array_equal(scaler.mean_, np.array([2, 3]))
        assert np.array_equal(scaler.scale_, np.array([1, 1]))

    def test_scaler_transform(self) -> None:
        """Test that transform method standardizes data."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        scaler.fit(X)
        result = scaler.transform(X)

        expected = np.array([[-1, -1], [1, 1]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_scaler_fit_transform(self) -> None:
        """Test fit_transform method."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        result = scaler.fit_transform(X)

        expected = np.array([[-1, -1], [1, 1]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_scaler_transform_without_fit(self) -> None:
        """Test that transform raises error if not fitted."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(PreprocessingError, match="not fitted"):
            scaler.transform(X)

    def test_handle_zeros_in_scale(self) -> None:
        """Test handling of zero variance features."""
        from ml_library.preprocessing.standard import _handle_zeros_in_scale

        scale = np.array([1.0, 0.0, 2.0])
        result = _handle_zeros_in_scale(scale)

        assert np.array_equal(result, np.array([1.0, 1.0, 2.0]))
        # Ensure original array is not modified
        assert np.array_equal(scale, np.array([1.0, 0.0, 2.0]))

    def test_scaler_with_zero_variance(self) -> None:
        """Test scaling with zero variance features."""
        scaler = StandardScaler()
        X = np.array([[1, 2, 3], [1, 4, 5]], dtype=np.float64)

        scaler.fit(X)
        result = scaler.transform(X)

        # First column has zero variance, so it should be scaled to zeros
        assert np.allclose(result[:, 0], [0, 0])
        # Other columns should be properly scaled
        assert not np.allclose(result[:, 1], [0, 0])

    def test_scaler_with_nan(self) -> None:
        """Test that scaler raises error for NaN data."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, np.nan]], dtype=np.float64)

        with pytest.raises(PreprocessingError, match="NaN or infinity"):
            scaler.fit(X)

    def test_scaler_transform_with_nan(self) -> None:
        """Test that transform raises error for NaN data."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        X_invalid = np.array([[1, np.inf], [3, 4]], dtype=np.float64)

        scaler.fit(X)
        with pytest.raises(PreprocessingError, match="NaN or infinity"):
            scaler.transform(X_invalid)

    def test_scaler_with_none_attributes(self) -> None:
        """Test that transform raises error if fitted but attributes are None."""
        scaler = StandardScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        # Manually set fitted but leave attributes as None
        scaler.fitted = True
        assert scaler.mean_ is None
        assert scaler.scale_ is None

        with pytest.raises(PreprocessingError, match="not properly fitted"):
            scaler.transform(X)


class TestMinMaxScaler:
    """Test the MinMaxScaler class."""

    def test_scaler_init(self) -> None:
        """Test that the MinMaxScaler initializes correctly."""
        scaler = MinMaxScaler()
        assert not scaler.fitted
        assert scaler.min_ is None
        assert scaler.scale_ is None
        assert scaler.data_min_ is None
        assert scaler.data_max_ is None
        assert scaler.data_range_ is None

    def test_scaler_fit(self) -> None:
        """Test that the MinMaxScaler fit method works."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        result = scaler.fit(X)

        assert scaler.fitted
        assert result is scaler
        assert np.array_equal(scaler.data_min_, np.array([1, 2]))
        assert np.array_equal(scaler.data_max_, np.array([3, 4]))
        assert np.array_equal(scaler.data_range_, np.array([2, 2]))
        # For feature range (0, 1), the formula is min_ = 0 - data_min * scale_
        # and scale_ = 1 / data_range_ for this case
        expected_min = np.array([0, 0]) - np.array([1, 2]) * (1.0 / np.array([2, 2]))
        assert np.allclose(scaler.min_, expected_min)
        assert np.allclose(scaler.scale_, np.array([0.5, 0.5]))

    def test_scaler_transform(self) -> None:
        """Test that transform method scales data to [0,1]."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        scaler.fit(X)
        result = scaler.transform(X)

        expected = np.array([[0, 0], [1, 1]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_scaler_fit_transform(self) -> None:
        """Test fit_transform method."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        result = scaler.fit_transform(X)

        expected = np.array([[0, 0], [1, 1]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_scaler_transform_without_fit(self) -> None:
        """Test that transform raises error if not fitted."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        with pytest.raises(PreprocessingError, match="not fitted"):
            scaler.transform(X)

    def test_scaler_with_custom_range(self) -> None:
        """Test scaling with custom range."""
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        result = scaler.fit_transform(X)

        expected = np.array([[-1, -1], [1, 1]], dtype=np.float64)
        assert np.allclose(result, expected)

    def test_scaler_with_constant_feature(self) -> None:
        """Test scaling with a constant feature."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [1, 4]], dtype=np.float64)

        result = scaler.fit_transform(X)

        # First column is constant, so it should be scaled to zeros
        assert np.allclose(result[:, 0], [0, 0])
        # Second column should be properly scaled
        assert np.allclose(result[:, 1], [0, 1])

    def test_scaler_with_nan(self) -> None:
        """Test that scaler raises error for NaN data."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, np.nan]], dtype=np.float64)

        with pytest.raises(PreprocessingError, match="NaN or infinity"):
            scaler.fit(X)

    def test_scaler_transform_with_inf(self) -> None:
        """Test that transform raises error for infinite data."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)
        X_invalid = np.array([[np.inf, 2], [3, 4]], dtype=np.float64)

        scaler.fit(X)
        with pytest.raises(PreprocessingError, match="NaN or infinity"):
            scaler.transform(X_invalid)

    def test_scaler_with_none_attributes(self) -> None:
        """Test that transform raises error if fitted but attributes are None."""
        scaler = MinMaxScaler()
        X = np.array([[1, 2], [3, 4]], dtype=np.float64)

        # Manually set fitted but leave attributes as None
        scaler.fitted = True
        assert scaler.scale_ is None
        assert scaler.min_ is None

        with pytest.raises(PreprocessingError, match="not properly fitted"):
            scaler.transform(X)
