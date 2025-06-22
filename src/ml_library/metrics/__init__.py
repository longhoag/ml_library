"""Metrics for evaluating model performance."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

__all__ = ["accuracy", "precision", "recall", "f1", "roc_auc", "mse", "mae", "r2"]


def accuracy(y_true, y_pred):
    """Calculate accuracy score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    score : float
        Accuracy score.
    """
    return accuracy_score(y_true, y_pred)


def precision(y_true, y_pred, average="weighted"):
    """Calculate precision score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy for multiclass problems.

    Returns
    -------
    score : float
        Precision score.
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def recall(y_true, y_pred, average="weighted"):
    """Calculate recall score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy for multiclass problems.

    Returns
    -------
    score : float
        Recall score.
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def f1(y_true, y_pred, average="weighted"):
    """Calculate F1 score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    average : str, default='weighted'
        Averaging strategy for multiclass problems.

    Returns
    -------
    score : float
        F1 score.
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def roc_auc(y_true, y_score, multi_class="ovr"):
    """Calculate ROC AUC score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_score : array-like
        Predicted probabilities.
    multi_class : str, default='ovr'
        Multiclass strategy.

    Returns
    -------
    score : float
        ROC AUC score.
    """
    try:
        return roc_auc_score(y_true, y_score, multi_class=multi_class)
    except (ValueError, TypeError):
        # Handle errors that can occur with inappropriate inputs
        return np.nan


def mse(y_true, y_pred):
    """Calculate mean squared error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    score : float
        Mean squared error.
    """
    return mean_squared_error(y_true, y_pred)


def mae(y_true, y_pred):
    """Calculate mean absolute error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    score : float
        Mean absolute error.
    """
    return mean_absolute_error(y_true, y_pred)


def r2(y_true, y_pred):
    """Calculate R² score.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    score : float
        R² score.
    """
    return r2_score(y_true, y_pred)
