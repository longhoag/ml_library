"""Ensemble model implementations."""

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the Random Forest classifier model."""
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Self:
        """Train the random forest model.

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
        self : RandomForestModel
            The trained model.
        """
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions using the random forest model.

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
        """
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        result = self.model.predict(X)
        return np.asarray(result)

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
        """
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        result = self.model.predict_proba(X)
        return np.asarray(result)

    def evaluate(
        self, X: NDArray[Any], y: NDArray[Any], metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the random forest model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.
        metrics : dict, optional
            Dictionary of additional metric functions.

        Returns
        -------
        metrics : dict
            Dictionary containing various evaluation metrics.
        """
        if not self.fitted:
            raise ValueError("Model must be trained before evaluation")
        y_pred = self.predict(X)

        # Calculate multiple metrics
        metrics_dict = {"accuracy": float(accuracy_score(y, y_pred))}

        # Handle binary vs. multi-class classification
        if len(np.unique(y)) <= 2:
            metrics_dict["precision"] = float(
                precision_score(y, y_pred, zero_division=0)
            )
            metrics_dict["recall"] = float(recall_score(y, y_pred, zero_division=0))
            metrics_dict["f1"] = float(f1_score(y, y_pred, zero_division=0))

            # Add roc_auc score for binary classification
            try:
                y_proba = self.predict_proba(X)
                # For binary classification, need probability of positive class
                if y_proba.shape[1] == 2:
                    from sklearn.metrics import roc_auc_score

                    metrics_dict["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
                else:
                    metrics_dict["roc_auc"] = float(
                        0.5
                    )  # Default value when not applicable
            except (ValueError, IndexError):
                # Handle cases where probabilities can't be computed or input is invalid
                metrics_dict["roc_auc"] = float(
                    0.5
                )  # Default value if calculation fails
        else:
            metrics_dict["precision"] = float(
                precision_score(y, y_pred, average="macro", zero_division=0)
            )
            metrics_dict["recall"] = float(
                recall_score(y, y_pred, average="macro", zero_division=0)
            )
            metrics_dict["f1"] = float(
                f1_score(y, y_pred, average="macro", zero_division=0)
            )

            # Add roc_auc score for multi-class classification
            try:
                y_proba = self.predict_proba(X)
                from sklearn.metrics import roc_auc_score

                metrics_dict["roc_auc"] = float(
                    roc_auc_score(y, y_proba, multi_class="ovr", average="macro")
                )
            except (ValueError, AttributeError):
                # Handle cases where probabilities can't be computed
                # Default value if calculation fails
                metrics_dict["roc_auc"] = float(0.5)

        # Update with additional metrics if provided
        if metrics:
            for name, func in metrics.items():
                metrics_dict[name] = float(func(y, y_pred))

        return metrics_dict

    @property
    def feature_importances_(self) -> NDArray[Any]:
        """Get feature importances."""
        if not self.fitted:
            raise ValueError("Model must be trained first")
        return np.asarray(self.model.feature_importances_)

    @property
    def classes_(self) -> NDArray[Any]:
        """Get class labels."""
        if not self.fitted:
            raise ValueError("Model must be trained first")
        return np.asarray(self.model.classes_)


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

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialize the Random Forest regressor model."""
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def train(self, X: NDArray[Any], y: NDArray[Any], **kwargs: Any) -> Self:
        """Train the random forest regressor model.

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
        self : RandomForestRegressorModel
            The trained model.
        """
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, X: NDArray[Any], **kwargs: Any) -> NDArray[Any]:
        """Make predictions using the random forest regressor model.

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
        """
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        result = self.model.predict(X)
        return np.asarray(result)

    def evaluate(
        self, X: NDArray[Any], y: NDArray[Any], metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Evaluate the random forest regressor model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.
        y : array-like of shape (n_samples,)
            True target values.
        metrics : dict, optional
            Dictionary of additional metric functions.

        Returns
        -------
        metrics : dict
            Dictionary containing various evaluation metrics.
        """
        if not self.fitted:
            raise ValueError("Model must be trained before evaluation")
        y_pred = self.predict(X)

        # Calculate standard regression metrics
        metrics_dict = {
            "r2": float(r2_score(y, y_pred)),
            "mse": float(mean_squared_error(y, y_pred)),
            "mae": float(mean_absolute_error(y, y_pred)),
        }

        # Update with additional metrics if provided
        if metrics:
            for name, func in metrics.items():
                metrics_dict[name] = float(func(y, y_pred))

        return metrics_dict

    @property
    def feature_importances_(self) -> NDArray[Any]:
        """Get feature importances."""
        if not self.fitted:
            raise ValueError("Model must be trained first")
        return np.asarray(self.model.feature_importances_)
