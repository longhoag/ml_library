"""Test basic functionality."""

from ml_library import Preprocessor


def test_preprocessor_init() -> None:
    """Test that preprocessor initializes correctly."""
    preprocessor = Preprocessor()
    assert preprocessor.fitted is False


def test_preprocessor_fit() -> None:
    """Test that preprocessor fit method works."""
    import numpy as np

    preprocessor = Preprocessor()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    result = preprocessor.fit(X, y)

    assert preprocessor.fitted is True
    assert result is preprocessor


def test_model_init() -> None:
    """Test that model initializes correctly."""
    # Model is abstract base class, can't be instantiated directly
    # So we use a concrete subclass or patching for testing
    from ml_library.models.linear import LinearModel

    model = LinearModel()
    assert model.fitted is False


def test_model_train() -> None:
    """Test that model train method works."""
    # Model is abstract base class, can't be instantiated directly
    # So we use a concrete subclass for testing
    import numpy as np

    from ml_library.models.linear import LinearModel

    model = LinearModel()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    result = model.train(X, y)

    assert model.fitted is True
    assert result is model
