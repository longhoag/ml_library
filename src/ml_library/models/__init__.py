"""Model definitions and utilities."""

import joblib

__all__ = [
    "Model",
    "LinearModel",
    "LogisticModel",
    "RandomForestModel",
    "RandomForestRegressorModel",
]


class Model:
    """Base class for all models."""

    def __init__(self):
        """Initialize the model."""
        self.trained = False
        self.model = None

    def train(self, X, y):
        """Train the model.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.

        Returns
        -------
        self : Model
            The trained model.
        """
        # Implementation will be provided in subclasses
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X : array-like
            Data to make predictions on.

        Returns
        -------
        y_pred : array-like
            Predicted values.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        # Implementation will be provided in subclasses
        return None

    def evaluate(self, X, y):
        """Evaluate the model.

        Parameters
        ----------
        X : array-like
            Test data.
        y : array-like
            True target values.

        Returns
        -------
        score : float
            Model evaluation score.
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        # Implementation will be provided in subclasses
        return 0.0

    def save(self, filepath):
        """Save the model to a file.

        Parameters
        ----------
        filepath : str
            Path to the file.
        """
        if not self.trained:
            raise ValueError("Model must be trained before saving")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Load a model from a file.

        Parameters
        ----------
        filepath : str
            Path to the file.

        Returns
        -------
        model : Model
            The loaded model.
        """
        return joblib.load(filepath)


from ml_library.models.ensemble import RandomForestModel, RandomForestRegressorModel

# Import specific model implementations
from ml_library.models.linear import LinearModel, LogisticModel
