"""Tests for the preprocessing module."""
import numpy as np
import pytest

from ml_library.preprocessing import (
    FeatureSelector,
    PolynomialPreprocessor,
    Preprocessor,
    StandardPreprocessor,
)


class TestBasePreprocessor:
    """Test the base Preprocessor class."""

    def test_preprocessor_init(self):
        """Test that the preprocessor initializes correctly."""
        preprocessor = Preprocessor()
        assert preprocessor.fitted is False

    def test_preprocessor_fit(self):
        """Test that the preprocessor fit method works."""
        preprocessor = Preprocessor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        result = preprocessor.fit(X, y)
        
        assert preprocessor.fitted is True
        assert result is preprocessor

    def test_preprocessor_transform_not_implemented(self):
        """Test that transform method raises NotImplementedError."""
        preprocessor = Preprocessor()
        X = np.array([[1, 2], [3, 4]])
        
        preprocessor.fit(X, np.array([0, 1]))
        
        with pytest.raises(NotImplementedError):
            preprocessor.transform(X)

    def test_preprocessor_fit_transform(self):
        """Test that fit_transform calls fit and transform."""
        class MockPreprocessor(Preprocessor):
            def transform(self, X):
                return np.array([[10, 20], [30, 40]])
        
        preprocessor = MockPreprocessor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        result = preprocessor.fit_transform(X, y)
        
        assert preprocessor.fitted is True
        assert np.array_equal(result, np.array([[10, 20], [30, 40]]))


class TestStandardPreprocessor:
    """Test the StandardPreprocessor class."""

    def test_preprocessor_init(self):
        """Test that the StandardPreprocessor initializes correctly."""
        preprocessor = StandardPreprocessor()
        assert preprocessor.fitted is False
        assert preprocessor.scaler is not None

    def test_preprocessor_fit(self):
        """Test that the StandardPreprocessor fit method works."""
        preprocessor = StandardPreprocessor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        result = preprocessor.fit(X, y)
        
        assert preprocessor.fitted is True
        assert result is preprocessor

    def test_preprocessor_transform(self):
        """Test that transform method standardizes data."""
        preprocessor = StandardPreprocessor()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)
        
        # Check that the result has zero mean and unit variance (approximately)
        assert np.abs(result.mean()) < 1e-10
        assert np.abs(result.std() - 1.0) < 1e-10


class TestPolynomialPreprocessor:
    """Test the PolynomialPreprocessor class."""

    def test_preprocessor_init(self):
        """Test that the PolynomialPreprocessor initializes correctly."""
        preprocessor = PolynomialPreprocessor(degree=2)
        assert preprocessor.fitted is False
        assert preprocessor.poly is not None
        assert preprocessor.degree == 2

    def test_preprocessor_fit(self):
        """Test that the PolynomialPreprocessor fit method works."""
        preprocessor = PolynomialPreprocessor(degree=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        result = preprocessor.fit(X, y)
        
        assert preprocessor.fitted is True
        assert result is preprocessor

    def test_preprocessor_transform(self):
        """Test that transform method adds polynomial features."""
        preprocessor = PolynomialPreprocessor(degree=2)
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])
        
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)
        
        # For degree=2 and 2 features, we should get 6 features:
        # [1, x1, x2, x1^2, x1*x2, x2^2]
        assert result.shape == (2, 6)


class TestFeatureSelector:
    """Test the FeatureSelector class."""

    def test_preprocessor_init(self):
        """Test that the FeatureSelector initializes correctly."""
        preprocessor = FeatureSelector(k=1)
        assert preprocessor.fitted is False
        assert preprocessor.k == 1
        assert preprocessor.feature_indices is None

    def test_preprocessor_fit(self):
        """Test that the FeatureSelector fit method works."""
        preprocessor = FeatureSelector(k=1)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        result = preprocessor.fit(X, y)
        
        assert preprocessor.fitted is True
        assert result is preprocessor
        assert preprocessor.feature_indices is not None
        assert len(preprocessor.feature_indices) == 1

    def test_preprocessor_transform(self):
        """Test that transform method selects top k features."""
        preprocessor = FeatureSelector(k=1)
        X = np.array([[1, 20], [3, 40], [5, 60]])
        y = np.array([0, 1, 0])
        
        preprocessor.fit(X, y)
        result = preprocessor.transform(X)
        
        # Should select the second column (index 1) because it has higher variance
        assert result.shape == (3, 1)
        assert np.array_equal(result, np.array([[20], [40], [60]]))
