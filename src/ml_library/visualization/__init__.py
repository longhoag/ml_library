"""Visualization utilities."""

from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from sklearn.model_selection import learning_curve

__all__ = ["plot_learning_curve", "plot_learning_curves"]


def plot_learning_curve(
    model: Any, X: np.ndarray, y: np.ndarray, cv: int = 5, n_jobs: Optional[int] = None
) -> Figure:
    """Plot learning curve for a model.

    Parameters
    ----------
    model : estimator
        A scikit-learn estimator.
    X : array-like
        Training data.
    y : array-like
        Target values.
    cv : int, default=5
        Number of cross-validation folds.
    n_jobs : int, optional
        Number of jobs to run in parallel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    train_sizes = np.linspace(0.1, 1.0, 10)
    # learning_curve returns train_sizes_abs, train_scores, test_scores
    # and if return_times=True, it also returns fit_times and score_times
    results = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_sizes, train_scores, test_scores = results[0], results[1], results[2]

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plot learning curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid()
    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="r",
    )
    ax.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
    )
    ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    ax.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")
    ax.legend(loc="best")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    return fig


def plot_learning_curves(
    train_scores: Sequence[float],
    val_scores: Sequence[float],
    metric_name: str = "Score",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot learning curves showing training and validation scores.

    Args:
        train_scores: Training scores per epoch/iteration.
        val_scores: Validation scores per epoch/iteration.
        metric_name: Name of the metric being plotted.
        title: Optional title for the plot.
        save_path: Optional path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)

    plt.plot(epochs, train_scores, "b-", label="Training " + metric_name)
    plt.plot(epochs, val_scores, "r-", label="Validation " + metric_name)

    plt.title(title or f"Learning Curves ({metric_name})")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
