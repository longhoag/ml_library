"""Test utility functions."""
import unittest
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ml_library.utils import train_test_split


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self) -> None:
        """Set up test cases."""
        # Create random data
        np.random.seed(42)
        n_samples = 1000
        n_features = 5
        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(2, size=n_samples)

    def test_train_test_split_shapes(self) -> None:
        """Test output shapes from train_test_split."""
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )

        expected_test_samples = int(len(self.X) * test_size)
        expected_train_samples = len(self.X) - expected_test_samples

        self.assertEqual(len(X_train), expected_train_samples)
        self.assertEqual(len(X_test), expected_test_samples)
        self.assertEqual(len(y_train), expected_train_samples)
        self.assertEqual(len(y_test), expected_test_samples)

    def test_train_test_split_random_state(self) -> None:
        """Test reproducibility with random_state."""
        random_state = 42
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=random_state
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=random_state
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)

    def test_train_test_split_different_random_states(self) -> None:
        """Test different splits with different random states."""
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=43
        )

        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(X_train1, X_train2)
