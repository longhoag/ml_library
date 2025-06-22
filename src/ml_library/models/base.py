"""Base model implementation."""

import pickle
from abc import ABC, abstractmethod

__all__ = ["Model"]


class Model(ABC):
    """Abstract base class for all models.

    This class defines the common interface that all models in the library
    must implement.

    All models must implement the following methods:
    - train: Train the model on data
    - predict: Make predictions using the trained model
    - evaluate: Evaluate the model performance on test data

    Models can optionally implement:
    - save: Save the model to disk
    - load: Load a model from disk
    """

    def __init__(self):
        """Initialize a new model."""
        self.trained = False

    @abstractmethod
    def train(self, X, y):
        """Train the model on the provided data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : Model
            The trained model.
        """

    @abstractmethod
    def predict(self, X):
        """Make predictions using the trained model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """

    @abstractmethod
    def evaluate(self, X, y):
        """Evaluate model performance on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            Performance score.
        """

    def save(self, path):
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Path to save the model to.

        Returns
        -------
        self : Model
            The model instance.
        """
        if not self.trained:
            raise ValueError("Cannot save untrained model")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return self

    @classmethod
    def load(cls, path):
        """Load a saved model from disk.

        Parameters
        ----------
        path : str
            Path to load the model from.

        Returns
        -------
        model : Model
            The loaded model.
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
