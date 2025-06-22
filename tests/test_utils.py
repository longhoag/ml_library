"""Tests for the utils module."""
import numpy as np
import pytest

from ml_library.utils import check_data, cross_validate, train_test_split


def test_check_data_valid():
    """Test check_data with valid inputs."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    X_checked, y_checked = check_data(X, y)

    assert np.array_equal(X, X_checked)
    assert np.array_equal(y, y_checked)


def test_check_data_list_inputs():
    """Test check_data with list inputs."""
    X = [[1, 2], [3, 4]]
    y = [0, 1]

    X_checked, y_checked = check_data(X, y)

    assert isinstance(X_checked, np.ndarray)
    assert isinstance(y_checked, np.ndarray)
    assert np.array_equal(X_checked, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(y_checked, np.array([0, 1]))


def test_check_data_mismatched_lengths():
    """Test check_data with mismatched lengths."""
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1]

    with pytest.raises(
        ValueError, match="X and y must have the same number of samples"
    ):
        check_data(X, y)


def test_train_test_split():
    """Test train_test_split with default test_size."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Default test_size is 0.2, so 20% of 4 samples = 1 test sample
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1
    assert y_train.shape[0] == 3
    assert y_test.shape[0] == 1


def test_train_test_split_custom_size():
    """Test train_test_split with custom test_size."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 2
    assert y_train.shape[0] == 2
    assert y_test.shape[0] == 2


def test_train_test_split_random_state():
    """Test train_test_split with fixed random_state."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    split1 = train_test_split(X, y, random_state=42)
    split2 = train_test_split(X, y, random_state=42)

    assert np.array_equal(split1[0], split2[0])  # X_train
    assert np.array_equal(split1[1], split2[1])  # X_test
    assert np.array_equal(split1[2], split2[2])  # y_train
    assert np.array_equal(split1[3], split2[3])  # y_test


class MockModel:
    """Mock model for testing cross_validate."""

    def __init__(self):
        self.train_calls = []
        self.evaluate_calls = []

    def train(self, X, y):
        self.train_calls.append((X, y))
        return self

    def evaluate(self, X, y):
        self.evaluate_calls.append((X, y))
        return {"accuracy": 0.9}


def test_cross_validate():
    """Test cross_validate with mock model."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 1, 0, 1, 0, 1])
    model = MockModel()

    metrics = cross_validate(model, X, y, cv=3)

    assert len(metrics) == 3  # 3-fold cross-validation
    assert "accuracy" in metrics[0]
    assert len(model.train_calls) == 3
    assert len(model.evaluate_calls) == 3
