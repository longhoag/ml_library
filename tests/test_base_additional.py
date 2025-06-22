"""Additional tests for model base class to improve coverage."""

import os
import tempfile
import unittest

import numpy as np
import pytest
from numpy.typing import NDArray

from ml_library.exceptions import NotFittedError
from ml_library.models.base import Model


class TestModelBase(unittest.TestCase):
    """Test for the base Model class functionality."""

    def test_base_model_not_implemented_methods(self):
        """Test that base model class raises NotImplementedError."""
        model = Model()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        with self.assertRaises(NotImplementedError):
            model.train(X, y)

        # Should raise NotFittedError first
        with self.assertRaises(NotFittedError):
            model.predict(X)

        # Simulate fitted model to test NotImplementedError in predict
        model.fitted = True
        with self.assertRaises(NotImplementedError):
            model.predict(X)

    def test_evaluate_not_fitted(self):
        """Test evaluate method when model is not fitted."""
        model = Model()
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        with self.assertRaises(NotFittedError):
            model.evaluate(X, y)

    def test_evaluate_default_behavior(self):
        """Test default evaluate implementation."""
        model = Model()
        model.fitted = True  # Simulate fitted model
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1])

        metrics = model.evaluate(X, y)
        self.assertEqual(metrics, {})  # Should return empty dict by default

    def test_save_and_load(self):
        """Test save and load methods of base model."""
        model = Model()
        model.fitted = True  # Need to set this manually for testing
        model.some_attr = "test_value"  # Add a custom attribute for testing

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as temp:
            temp_filename = temp.name

        try:
            # Test save method
            saved_model = model.save(temp_filename)
            self.assertIs(saved_model, model)  # Should return self
            self.assertTrue(os.path.exists(temp_filename))
            self.assertTrue(os.path.getsize(temp_filename) > 0)

            # Test load method
            new_model = Model().load(temp_filename)
            self.assertTrue(new_model.fitted)
            self.assertEqual(new_model.some_attr, "test_value")
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class MockModel(Model):
    """Mock model implementation for testing Model base class."""
    
    def train(self, X: NDArray, y: NDArray, **kwargs) -> "MockModel":
        """Mock implementation of train."""
        self.fitted = True
        return self
    
    def predict(self, X: NDArray, **kwargs) -> NDArray:
        """Mock implementation of predict."""
        if not self.fitted:
            raise NotFittedError("Model not fitted")
        return np.zeros(len(X))


def test_mock_model():
    """Test Model class with a concrete implementation."""
    model = MockModel()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    
    # Test train method
    returned_model = model.train(X, y)
    assert returned_model is model
    assert model.fitted is True
    
    # Test predict method
    pred = model.predict(X)
    assert isinstance(pred, np.ndarray)
    assert len(pred) == len(X)


def test_str_and_repr():
    """Test string representation of Model objects."""
    model = Model()
    str_rep = str(model)
    repr_rep = repr(model)
    
    assert "Model" in str_rep
    assert "Model" in repr_rep
