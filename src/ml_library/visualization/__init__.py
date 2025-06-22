"""Visualization utilities for the ML library."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

__all__ = ["plot_learning_curve"]


def plot_learning_curve(model, X, y, cv=5, n_jobs=None):
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
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )

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
