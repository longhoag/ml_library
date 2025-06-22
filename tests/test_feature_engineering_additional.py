"""Additional tests for feature engineering module to improve coverage."""

import numpy as np
import pytest

from ml_library.exceptions import PreprocessingError
from ml_library.preprocessing.feature_engineering import (
    FeatureSelector,
    PolynomialPreprocessor,
)


class TestFeatureSelectorEdgeCases:
    """Tests for FeatureSelector edge cases."""
    
    def test_empty_input(self):
        """Test handling of empty input."""
        selector = FeatureSelector(k=2)
        
        # Empty 2D array should raise ValueError during fit
        X_empty = np.array([]).reshape(0, 0)
        with pytest.raises(ValueError):
            selector.fit(X_empty)
    
    def test_k_greater_than_features(self):
        """Test when k is greater than number of features."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        # k=4 is greater than number of features (3)
        selector = FeatureSelector(k=4)
        with pytest.raises(ValueError):
            selector.fit(X)
    
    def test_invalid_feature_indices(self):
        """Test with invalid feature indices."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        
        # Feature index 5 is out of range (max valid index is 2)
        selector = FeatureSelector(feature_indices=[0, 5])
        with pytest.raises(ValueError):
            selector.fit(X)
    
    def test_get_support_not_fitted(self):
        """Test get_support property when not fitted."""
        selector = FeatureSelector(k=2)
        
        with pytest.raises(PreprocessingError):
            _ = selector.get_support
    
    def test_transform_shape_mismatch(self):
        """Test transform with shape mismatch."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        selector = FeatureSelector(k=2)
        selector.fit(X)
        
        # Try transforming data with wrong feature count
        X_wrong = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            selector.transform(X_wrong)
    
    def test_transform_not_fitted(self):
        """Test transform without fitting first."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        selector = FeatureSelector(k=2)
        
        with pytest.raises(PreprocessingError):
            selector.transform(X)
    
    def test_fit_exception_handling(self):
        """Test exception handling in fit method."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        selector = FeatureSelector(k=2)
        
        # Create a mock object that raises an exception during variance calculation
        original_var = np.var
        try:
            # Replace np.var with a function that raises Exception
            np.var = lambda *args, **kwargs: exec('raise RuntimeError("Test error")')
            
            with pytest.raises(PreprocessingError):
                selector.fit(X)
        finally:
            # Restore original function
            np.var = original_var


class TestPolynomialPreprocessorEdgeCases:
    """Tests for PolynomialPreprocessor edge cases."""
    
    def test_transform_not_fitted(self):
        """Test transform without fitting first."""
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialPreprocessor(degree=2)
        
        with pytest.raises(PreprocessingError):
            poly.transform(X)
    
    def test_fit_transform_error_handling(self):
        """Test error handling in fit_transform method."""
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialPreprocessor(degree=2)
        
        # Use monkey patching to simulate a problematic transformer
        from types import MethodType
        
        def mock_fit(self, X, y=None):
            raise ValueError("Simulated fit error")
            
        # Apply mock to transformer
        poly.transformer.fit = MethodType(mock_fit, poly.transformer)
        
        with pytest.raises(PreprocessingError):
            poly.fit_transform(X)
            
        # Clean up by creating a new instance
        poly = PolynomialPreprocessor(degree=2)
    
    def test_transform_error_handling(self):
        """Test error handling in transform method."""
        X = np.array([[1, 2], [3, 4]])
        poly = PolynomialPreprocessor(degree=2)
        poly.fit(X)
        
        # Use monkey patching to simulate a problematic transformer
        from types import MethodType
        
        def mock_transform(self, X):
            raise ValueError("Simulated transform error")
            
        # Apply mock to transformer
        poly.transformer.transform = MethodType(mock_transform, poly.transformer)
        
        with pytest.raises(PreprocessingError):
            poly.transform(X)
