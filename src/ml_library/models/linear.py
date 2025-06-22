"""Linear model implementations."""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

from ml_library.models import Model

__all__ = ["LinearModel", "LogisticModel"]


class LinearModel(Model):
    """Linear regression model.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, fit_intercept=True):
        """Initialize the linear model."""
        super().__init__()
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def train(self, X, y):
        """Train the linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LinearModel
            The trained model.
        """
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions using the linear regression model.

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
        """Evaluate the linear regression model.

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
    def coef_(self):
        """Get coefficients of the model."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.coef_

    @property
    def intercept_(self):
        """Get intercept of the model."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.intercept_


class LogisticModel(Model):
    """Logistic regression model for classification.

    Parameters
    ----------
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=100
        Maximum number of iterations.
    """

    def __init__(self, C=1.0, max_iter=100):
        """Initialize the logistic model."""
        super().__init__()
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(C=C, max_iter=max_iter)

    def train(self, X, y):
        """Train the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : LogisticModel
            The trained model.
        """
        self.model.fit(X, y)
        self.trained = True
        return self

    def predict(self, X):
        """Make predictions using the logistic regression model.

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
        """Evaluate the logistic regression model.

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
    def coef_(self):
        """Get coefficients of the model."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.coef_

    @property
    def intercept_(self):
        """Get intercept of the model."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.intercept_

    @property
    def classes_(self):
        """Get class labels."""
        if not self.trained:
            raise ValueError("Model must be trained first")
        return self.model.classes_
