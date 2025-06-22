"""Model definitions and utilities."""

# Import specific model implementations - these need to be imported here
# for proper package structure but will be imported at the bottom
# to avoid circular imports
from typing import ClassVar, Optional, TypeVar, Union, cast

import joblib
import numpy as np  # noqa: F401
from typing_extensions import Self

T = TypeVar("T", bound="Model")

__all__ = [
    "Model",
    "LinearModel",
    "LogisticModel",
    "RandomForestModel",
    "RandomForestRegressorModel",
]


class Model:
    """Base class for all models."""

    def __init__(self) -> None:
        """Initialize the model."""
        self.trained = False
        self.model: Optional[object] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> Self:
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

    def predict(self, X: np.ndarray) -> np.ndarray:
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
        return np.array([])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
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

    def save(self, filepath: str) -> None:
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
    def load(cls: ClassVar[T], filepath: str) -> T:
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
        return cast(T, joblib.load(filepath))


from ml_library.models.ensemble import RandomForestModel, RandomForestRegressorModel

# Import specific implementations at the end to avoid circular imports
from ml_library.models.linear import LinearModel, LogisticModel
