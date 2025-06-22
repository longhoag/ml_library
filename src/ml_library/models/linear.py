"""Linear model implementations."""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from typing_extensions import Self

from ml_library.exceptions import DataError, NotFittedError
from ml_library.logging import get_logger
from ml_library.models import Model

__all__ = ["LinearModel", "LogisticModel"]

# Setup logger for this module
logger = get_logger(__name__)


class LinearModel(Model):
    """Linear regression model.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        """Initialize the linear model."""
        super().__init__()
        self.fit_intercept = fit_intercept
        self.model = LinearRegression(fit_intercept=fit_intercept)

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Self:
        """Train the linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional training parameters.

        Returns
        -------
        self : LinearModel
            The trained model.

        Raises
        ------
        DataError
            If there is an issue with the input data.
        """
        try:
            logger.info("Training LinearModel with data of shape %s", str(np.shape(X)))
            self.model.fit(X, y)
            self.fitted = True
            logger.debug("LinearModel successfully trained")
            return self
        except Exception as e:
            logger.exception("Error training LinearModel: %s", str(e))
            raise DataError(
                f"Error training LinearModel: {str(e)}", data_shape=np.shape(X)
            ) from e

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions using the linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.
        **kwargs : dict
            Additional prediction parameters.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        DataError
            If there is an issue with the input data.
        """
        if not self.fitted:
            logger.error("Attempted to predict with untrained LinearModel")
            raise NotFittedError(
                "Model must be trained before prediction", model_type="LinearModel"
            )

        try:
            logger.debug(
                "Making predictions with LinearModel on data of shape %s",
                str(np.shape(X)),
            )
            result = self.model.predict(X)
            return np.asarray(result)
        except Exception as e:
            logger.exception("Error predicting with LinearModel: %s", str(e))
            raise DataError(
                f"Error predicting: {str(e)}", data_shape=np.shape(X)
            ) from e

    def evaluate(
        self, X: NDArray[Any], y: NDArray[Any], metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the linear regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.
        metrics : dict, optional
            Additional evaluation metrics.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing evaluation metrics.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        DataError
            If there is an issue with the input data.
        """
        if not self.fitted:
            logger.error("Attempted to evaluate untrained LinearModel")
            raise NotFittedError(
                "Model must be trained before evaluation", model_type="LinearModel"
            )

        try:
            y_pred = self.predict(X)
            metrics_dict = {
                "r2": float(r2_score(y, y_pred)),
                "mse": float(mean_squared_error(y, y_pred)),
                "mae": float(mean_absolute_error(y, y_pred)),
            }

            # Add user-provided metrics if any
            if metrics:
                for name, metric_func in metrics.items():
                    metrics_dict[name] = float(metric_func(y, y_pred))

            logger.debug("Evaluation metrics: %s", str(metrics_dict))
            return metrics_dict
        except Exception as e:
            logger.exception("Error evaluating LinearModel: %s", str(e))
            raise DataError(
                f"Error evaluating: {str(e)}", data_shape=np.shape(X)
            ) from e

    @property
    def coef_(self) -> NDArray[Any]:
        """Get coefficients of the model."""
        if not self.fitted:
            logger.error("Attempted to access coefficients of untrained LinearModel")
            raise NotFittedError(
                "Model must be trained before accessing coefficients",
                model_type="LinearModel",
            )
        return np.asarray(self.model.coef_)

    @property
    def intercept_(self) -> Any:
        """Get intercept of the model."""
        if not self.fitted:
            logger.error("Attempted to access intercept of untrained LinearModel")
            raise NotFittedError(
                "Model must be trained before accessing intercept",
                model_type="LinearModel",
            )
        return self.model.intercept_


class LogisticModel(Model):
    """Logistic regression model.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    C : float, default=1.0
        Inverse of regularization strength.
    max_iter : int, default=100
        Maximum number of iterations for solver.
    """

    def __init__(
        self, fit_intercept: bool = True, C: float = 1.0, max_iter: int = 100
    ) -> None:
        """Initialize the logistic regression model."""
        super().__init__()
        self.fit_intercept = fit_intercept
        self.C = C
        self.max_iter = max_iter
        self.model = LogisticRegression(
            fit_intercept=fit_intercept, C=C, max_iter=max_iter
        )

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Self:
        """Train the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        **kwargs : dict
            Additional training parameters.

        Returns
        -------
        self : LogisticModel
            The trained model.

        Raises
        ------
        DataError
            If there is an issue with the input data.
        """
        try:
            logger.info(
                "Training LogisticModel with data of shape %s", str(np.shape(X))
            )
            self.model.fit(X, y)
            self.fitted = True
            logger.debug("LogisticModel successfully trained")
            return self
        except Exception as e:
            logger.exception("Error training LogisticModel: %s", str(e))
            raise DataError(
                f"Error training LogisticModel: {str(e)}", data_shape=np.shape(X)
            ) from e

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions using the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.
        **kwargs : dict
            Additional prediction parameters.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        DataError
            If there is an issue with the input data.
        """
        if not self.fitted:
            logger.error("Attempted to predict with untrained LogisticModel")
            raise NotFittedError(
                "Model must be trained before prediction", model_type="LogisticModel"
            )

        try:
            result = self.model.predict(X)
            return np.asarray(result)
        except Exception as e:
            logger.exception("Error predicting with LogisticModel: %s", str(e))
            raise DataError(
                f"Error predicting: {str(e)}", data_shape=np.shape(X)
            ) from e

    def predict_proba(self, X: NDArray[Any]) -> NDArray[Any]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to make predictions on.

        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        DataError
            If there is an issue with the input data.
        """
        if not self.fitted:
            logger.error(
                "Attempted to predict probabilities with untrained LogisticModel"
            )
            raise NotFittedError(
                "Model must be trained before prediction", model_type="LogisticModel"
            )

        try:
            result = self.model.predict_proba(X)
            return np.asarray(result)
        except Exception as e:
            logger.exception(
                "Error predicting probabilities with LogisticModel: %s", str(e)
            )
            raise DataError(
                f"Error predicting probabilities: {str(e)}", data_shape=np.shape(X)
            ) from e

    def evaluate(
        self, X: NDArray[Any], y: NDArray[Any], metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.
        metrics : dict, optional
            Additional evaluation metrics.

        Returns
        -------
        metrics_dict : dict
            Dictionary containing various evaluation metrics.

        Raises
        ------
        NotFittedError
            If the model has not been trained.
        DataError
            If there is an issue with the input data.
        """
        if not self.fitted:
            logger.error("Attempted to evaluate untrained LogisticModel")
            raise NotFittedError(
                "Model must be trained before evaluation", model_type="LogisticModel"
            )

        try:
            y_pred = self.predict(X)
            metrics_dict = {
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, average="macro")),
                "recall": float(recall_score(y, y_pred, average="macro")),
                "f1": float(f1_score(y, y_pred, average="macro")),
            }

            # Add user-provided metrics if any
            if metrics:
                for name, metric_func in metrics.items():
                    metrics_dict[name] = float(metric_func(y, y_pred))

            logger.debug("Evaluation metrics: %s", str(metrics_dict))
            return metrics_dict
        except Exception as e:
            logger.exception("Error evaluating LogisticModel: %s", str(e))
            raise DataError(
                f"Error evaluating: {str(e)}", data_shape=np.shape(X)
            ) from e

    @property
    def coef_(self) -> NDArray[Any]:
        """Get coefficients of the model."""
        if not self.fitted:
            logger.error("Attempted to access coefficients of untrained LogisticModel")
            raise NotFittedError(
                "Model must be trained before accessing coefficients",
                model_type="LogisticModel",
            )
        return np.asarray(self.model.coef_)

    @property
    def intercept_(self) -> NDArray[Any]:
        """Get intercept of the model."""
        if not self.fitted:
            logger.error("Attempted to access intercept of untrained LogisticModel")
            raise NotFittedError(
                "Model must be trained before accessing intercept",
                model_type="LogisticModel",
            )
        return np.asarray(self.model.intercept_)

    @property
    def classes_(self) -> NDArray[Any]:
        """Get class labels."""
        if not self.fitted:
            logger.error("Attempted to access classes of untrained LogisticModel")
            raise NotFittedError(
                "Model must be trained before accessing classes",
                model_type="LogisticModel",
            )
        return np.asarray(self.model.classes_)
