"""Visualization utilities for the ML library."""

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
    # Implementation will be added in Phase 2
    return None
