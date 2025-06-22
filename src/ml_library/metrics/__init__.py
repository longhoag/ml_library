"""Metric functions for model evaluation."""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "mse",
    "mae",
    "r2",
    "roc_auc",
]


def accuracy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Calculate accuracy score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy score.
    """
    return float(np.mean(y_true == y_pred))


def precision(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], zero_division: float = 0.0
) -> float:
    """Calculate precision score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    zero_division : float, default=0.0
        Value to return when there is a zero division.

    Returns
    -------
    float
        Precision score.
    """
    true_positives: float = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positives: float = np.sum(y_pred == 1)

    if predicted_positives == 0:
        return zero_division

    return float(true_positives / predicted_positives)


def recall(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], zero_division: float = 0.0
) -> float:
    """Calculate recall score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    zero_division : float, default=0.0
        Value to return when there is a zero division.

    Returns
    -------
    float
        Recall score.
    """
    true_positives: float = np.sum((y_true == 1) & (y_pred == 1))
    actual_positives: float = np.sum(y_true == 1)

    if actual_positives == 0:
        return zero_division

    return float(true_positives / actual_positives)


def f1(
    y_true: NDArray[np.float64], y_pred: NDArray[np.float64], zero_division: float = 0.0
) -> float:
    """Calculate F1 score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    zero_division : float, default=0.0
        Value to return when there is a zero division.

    Returns
    -------
    float
        F1 score.
    """
    prec = precision(y_true, y_pred, zero_division)
    rec = recall(y_true, y_pred, zero_division)

    if prec == 0 and rec == 0:
        return zero_division

    return float(2 * (prec * rec) / (prec + rec))


def mse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Calculate mean squared error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Calculate mean absolute error.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    """Calculate R2 score.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        R2 score.
    """
    ss_total: float = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual: float = np.sum((y_true - y_pred) ** 2)

    if ss_total == 0:
        return 0.0

    return float(1 - (ss_residual / ss_total))


def roc_auc(y_true: NDArray[np.float64], y_score: NDArray[np.float64]) -> float:
    """Calculate ROC AUC score.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_score : array-like
        Predicted probabilities or scores.

    Returns
    -------
    float
        ROC AUC score.
    """
    # Sort by predictions
    sorted_indices = np.argsort(y_score)[::-1]
    y_true = y_true[sorted_indices]

    n_pos: float = np.sum(y_true == 1)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Calculate true positive rate and false positive rate
    tpr = np.cumsum(y_true) / n_pos
    fpr = np.cumsum(1 - y_true) / n_neg

    # Add endpoints
    tpr = np.concatenate(([0], tpr, [1]))
    fpr = np.concatenate(([0], fpr, [1]))

    # Calculate AUC using trapezoidal rule
    # Use trapezoid instead of trapz (which is deprecated)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(tpr, fpr))
    else:
        # Fallback to trapz for older numpy versions
        return float(np.trapz(tpr, fpr))
