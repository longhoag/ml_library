"""Test visualization functions."""
import os
import unittest

from ml_library.visualization import plot_learning_curves


class TestVisualization(unittest.TestCase):
    """Test cases for visualization functions."""

    def setUp(self) -> None:
        """Set up test cases.

        Creates test data for learning curves.
        """
        self.train_scores = [0.8, 0.85, 0.9, 0.92, 0.95]
        self.val_scores = [0.75, 0.8, 0.82, 0.85, 0.86]
        self.test_output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        if not os.path.exists(self.test_output_dir):
            os.makedirs(self.test_output_dir)

    def test_plot_learning_curves(self) -> None:
        """Test plotting of learning curves.

        Verifies that learning curves can be plotted with different parameters.
        """
        plot_learning_curves(self.train_scores, self.val_scores, metric_name="Accuracy")

    def test_plot_learning_curves_with_save(self) -> None:
        """Test saving learning curves plot.

        Verifies that learning curves can be saved to a file.
        """
        save_path = os.path.join(self.test_output_dir, "learning_curves.png")
        plot_learning_curves(
            self.train_scores,
            self.val_scores,
            metric_name="Loss",
            title="Training Progress",
            save_path=save_path,
        )
        self.assertTrue(os.path.exists(save_path))
        # Clean up
        os.remove(save_path)

    def tearDown(self) -> None:
        """Clean up test outputs.

        Removes test output directory if it exists.
        """
        if os.path.exists(self.test_output_dir):
            for file in os.listdir(self.test_output_dir):
                os.remove(os.path.join(self.test_output_dir, file))
            os.rmdir(self.test_output_dir)
