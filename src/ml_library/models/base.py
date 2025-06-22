"""Base model implementation."""

import pickle
from abc import ABC, abstractmethod

from ml_library.exceptions import NotFittedError
from ml_library.logging import get_logger

__all__ = ["Model"]

# Setup logger for this module
logger = get_logger(__name__)


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
        logger.debug("Initialized %s model", self.__class__.__name__)

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
            
        Raises
        ------
        NotFittedError
            If the model has not been trained.
        IOError
            If there is an issue with file I/O.
        """
        if not self.trained:
            logger.error("Attempted to save untrained %s model", self.__class__.__name__)
            raise NotFittedError("Cannot save untrained model", model_type=self.__class__.__name__)
        
        try:
            logger.info("Saving %s model to %s", self.__class__.__name__, path)
            with open(path, "wb") as f:
                pickle.dump(self, f)
            logger.debug("Successfully saved model to %s", path)
        except IOError:
            logger.exception("Failed to save %s model to %s", self.__class__.__name__, path)
            raise
            
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
            
        Raises
        ------
        IOError
            If there is an issue with file I/O.
        ValueError
            If the loaded object is not of the correct type.
        """
        try:
            logger.info("Loading model from %s", path)
            with open(path, "rb") as f:
                model = pickle.load(f)
                
            if not isinstance(model, cls):
                logger.error("Loaded object is not a %s instance", cls.__name__)
                raise ValueError(f"Loaded object is not a {cls.__name__} instance")
                
            logger.debug("Successfully loaded %s from %s", model.__class__.__name__, path)
            return model
            
        except IOError:
            logger.exception("Failed to load model from %s", path)
            raise
