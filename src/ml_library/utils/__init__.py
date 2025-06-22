"""Utility functions for the ML library."""

import numpy as np
from sklearn.utils import check_array, check_consistent_length

from ml_library.exceptions import DataError
from ml_library.logging import get_logger

__all__ = ["check_data", "train_test_split", "cross_validate"]

# Setup logger for this module
logger = get_logger(__name__)


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
        
    Raises
    ------
    DataError
        If the data is invalid or inconsistent.
    """
    try:
        # Check X array
        logger.debug("Checking input data with shape: %s", str(np.shape(X)))
        X = check_array(X, ensure_2d=ensure_2d)

        # If y is provided, check y array and ensure consistent length
        if y is not None:
            logger.debug("Checking target data with shape: %s", str(np.shape(y)))
            y = np.asarray(y)
            try:
                check_consistent_length(X, y)
            except ValueError as e:
                msg = f"Inconsistent data shapes: X {np.shape(X)}, y {np.shape(y)}"
                logger.error(msg)
                raise DataError(msg, data_shape=(np.shape(X), np.shape(y))) from e
            logger.debug("Data check successful")
            return X, y
        
        logger.debug("Data check successful (X only)")
        return X
        
    except Exception as e:
        if not isinstance(e, DataError):
            logger.exception("Data check failed: %s", str(e))
            raise DataError(f"Invalid data format: {str(e)}", 
                            data_shape=getattr(X, "shape", None)) from e
        raise


def train_test_split(X, y=None, test_size=0.25, random_state=None, shuffle=True):
    """Split arrays into random train and test subsets.

    Parameters
    ----------
    X : array-like
        Data to split.
    y : array-like, optional
        Target values.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, default=None
        Controls the shuffling applied to the data before splitting.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.

    Returns
    -------
    X_train, X_test, y_train, y_test : array-like
        Train and test splits of data and target.
    """
    from sklearn.model_selection import train_test_split as sklearn_split

    if y is not None:
        return sklearn_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
    else:
        return sklearn_split(
            X, test_size=test_size, random_state=random_state, shuffle=shuffle
        )


def cross_validate(model, X, y, cv=5, scoring="accuracy"):
    """Evaluate model using cross-validation.

    Parameters
    ----------
    model : Model
        A ml_library model instance.
    X : array-like
        Training data.
    y : array-like
        Target values.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric to use.

    Returns
    -------
    scores : dict
        Dictionary containing the cross-validation scores.
    """
    from sklearn.model_selection import cross_validate as sklearn_cv

    # Clone our model to sklearn compatible format
    def _fit(X, y):
        model.train(X, y)
        return model

    scores = sklearn_cv(_fit, X, y, cv=cv, scoring=scoring, return_train_score=True)

    return scores
