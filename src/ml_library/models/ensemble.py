"""Ensemble model implementations."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

from ml_library.models import Model

__all__ = ["RandomForestModel", "RandomForestRegressorModel"]


class RandomForestModel(Model):
    """Random Forest classifier model.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of the trees.
    random_state : int, default=None
        Random state for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """Initialize the Random Forest classifier model."""
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, X, y):
        """Train the random forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestModel
            The trained model.
        """
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions using the random forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluate the random forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            Accuracy score.
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    @property
    def feature_importances_(self):
        """Get feature importances."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_

    @property
    def classes_(self):
        """Get class labels."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.classes_


class RandomForestRegressorModel(Model):
    """Random Forest regressor model.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, default=None
        Maximum depth of the trees.
    random_state : int, default=None
        Random state for reproducibility.
    """

    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        """Initialize the Random Forest regressor model."""
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, X, y):
        """Train the random forest regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RandomForestRegressorModel
            The trained model.
        """
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions using the random forest regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the random forest regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.

        Returns
        -------
        score : float
            R² score.
        """
        if not self.trained:
            raise ValueError("Model must be trained before evaluation")
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    @property
    def feature_importances_(self):
        """Get feature importances."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
