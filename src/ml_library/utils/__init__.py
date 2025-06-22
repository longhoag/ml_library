"""Utility functions for the ML library."""

__all__ = ["check_data"]


def check_data(X, y=None, ensure_2d=True):
    """Check the data for valid format.

    Parameters
    ----------
    X : array-like
        Data to check.
    y : array-like, optional
        Target values.
    ensure_2d : bool, default=True
        Whether to ensure X is 2D.

    Returns
    -------
    X : array-like
        Checked data.
    y : array-like, optional
        Checked target values if provided.
    """
    # Implementation will be added in Phase 2
    if y is not None:
        return X, y
    return X
