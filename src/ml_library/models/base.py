"""Base model implementation."""

from typing import Any, Dict, Optional

import joblib
import numpy as np
from numpy.typing import NDArray

from ml_library.exceptions import NotFittedError
from ml_library.logging import get_logger

__all__ = ["Model"]

# Setup logger for this module
logger = get_logger(__name__)


class Model:
    """Base class for all models in the library.

    This class defines the common interface that all ML models should implement.
    Subclasses should implement train(), predict(), and evaluate() methods.
    """

    def __init__(self) -> None:
        """Initialize a new model instance."""
        self.fitted = False
        self.logger = logger

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> "Model":
        """Train the model.

        Args:
            X: Training data.
            y: Target values.
            **kwargs: Additional training parameters.

        Returns:
            self: The trained model instance.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(
            f"train() not implemented in {self.__class__.__name__}"
        )

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions for input data.

        Args:
            X: Input data.
            **kwargs: Additional prediction parameters.

        Returns:
            Predicted values.

        Raises:
            NotFittedError: If model is not trained.
            NotImplementedError: If not implemented by subclass.
        """
        if not self.fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'train()' with appropriate arguments before using this model."
            )

        raise NotImplementedError(
            f"predict() not implemented in {self.__class__.__name__}"
        )

    def evaluate(
        self, X: NDArray[Any], y: NDArray[Any], metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X: Input data.
            y: True labels/values.
            metrics: Dictionary of metric functions.

        Returns:
            Dictionary with metric names and values.

        Raises:
            NotFittedError: If model is not trained.
        """
        if not self.fitted:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'train()' with appropriate arguments before evaluating."
            )

        # By default, return empty metrics
        return {}

    def save(self, filepath: str) -> "Model":
        """Save the model to a file.

        Args:
            filepath: Path where to save the model.

        Returns:
            self: The model instance for chaining.

        Raises:
            NotFittedError: If the model is not fitted.
        """
        if not self.fitted:
            raise NotFittedError("Cannot save model that has not been trained")

        joblib.dump(self, filepath)
        return self

    @classmethod
    def load(cls, filepath: str) -> "Model":
        """Load a model from a file.

        Args:
            filepath: Path to the saved model file.

        Returns:
            Model: The loaded model instance.

        Raises:
            ValueError: If the loaded object is not an instance of the calling class.
        """
        model = joblib.load(filepath)
        if not isinstance(model, cls):
            raise ValueError(f"Loaded object is not an instance of {cls.__name__}")
        return model
