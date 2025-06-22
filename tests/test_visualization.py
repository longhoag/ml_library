"""Test visualization functions."""
import os
import unittest
import numpy as np
from unittest.mock import patch, Mock
from sklearn.linear_model import LinearRegression

from ml_library.visualization import plot_learning_curve, plot_learning_curves


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

        # Create sample data for learning curve
        self.X = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
                [11, 12],
                [13, 14],
                [15, 16],
                [17, 18],
                [19, 20],
            ]
        )
        self.y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
        self.model = LinearRegression()
        
    @patch('ml_library.visualization.learning_curve')
    def test_plot_learning_curve(self, mock_learning_curve) -> None:
        """Test plotting of scikit-learn learning curve.
        
        Verifies that learning curve can be plotted for a model.
        """
        # Mock the learning_curve function
        train_sizes = np.linspace(0.1, 1.0, 5)
        train_scores = np.array([[0.6, 0.65], [0.7, 0.75], [0.8, 0.85], [0.9, 0.95], [0.95, 0.98]])
        test_scores = np.array([[0.55, 0.6], [0.65, 0.7], [0.75, 0.8], [0.85, 0.9], [0.9, 0.92]])
        
        # Configure mock to return our predefined values
        mock_learning_curve.return_value = (train_sizes, train_scores, test_scores)
        
        # Call the function
        fig = plot_learning_curve(self.model, self.X, self.y, cv=2)
        
        # Verify results
        self.assertIsNotNone(fig)
        # Assert that learning_curve was called
        mock_learning_curve.assert_called_once()

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
