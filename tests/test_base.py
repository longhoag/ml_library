"""Test basic functionality."""

import pytest

from ml_library import Model, Preprocessor


def test_preprocessor_init():
    """Test that preprocessor initializes correctly."""
    preprocessor = Preprocessor()
    assert preprocessor.fitted is False


def test_preprocessor_fit():
    """Test that preprocessor fit method works."""
    preprocessor = Preprocessor()
    X = [[1, 2], [3, 4]]
    y = [0, 1]
    
    result = preprocessor.fit(X, y)
    
    assert preprocessor.fitted is True
    assert result is preprocessor


def test_model_init():
    """Test that model initializes correctly."""
    model = Model()
    assert model.trained is False
    assert model.model is None


def test_model_train():
    """Test that model train method works."""
    model = Model()
    X = [[1, 2], [3, 4]]
    y = [0, 1]
    
    result = model.train(X, y)
    
    assert model.trained is True
    assert result is model
