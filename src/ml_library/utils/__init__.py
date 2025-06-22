"""Utility functions for the ML library."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split as sk_train_test_split
from sklearn.utils import check_array, check_consistent_length

from ml_library.exceptions import DataError
from ml_library.logging import get_logger

__all__ = ["check_data", "train_test_split", "cross_validate"]

# Setup logger for this module
logger = get_logger(__name__)


def check_data(
    X: NDArray[Any], y: Optional[NDArray[Any]] = None, ensure_2d: bool = True
) -> Tuple[NDArray[Any], Optional[NDArray[Any]]]:
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
        return X, None

    except Exception as e:
        if not isinstance(e, DataError):
            logger.exception("Data check failed: %s", str(e))
            raise DataError(
                f"Invalid data format: {str(e)}", data_shape=getattr(X, "shape", None)
            ) from e
        raise


def train_test_split(
    X: NDArray[Any],
    y: NDArray[Any],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
    """Split arrays or matrices into random train and test subsets.

    Args:
        X: The input samples.
        y: The target values.
        test_size: The proportion of data to include in the test split.
        random_state: Controls the randomness of the split.

    Returns:
        X_train, X_test, y_train, y_test: Train and test splits of X and y.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_idx, train_idx = indices[:n_test], indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def cross_validate(
    model: Any, X: NDArray[Any], y: NDArray[Any], cv: int = 5, scoring: str = "accuracy"
) -> Dict[str, List[float]]:
    """Evaluate model using cross-validation.

    Parameters
    ----------
    model : Model or BaseEstimator
        A ml_library model instance or a scikit-learn compatible estimator.
    X : array-like
        Training data.
    y : array-like
        Target values.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str or list, default='accuracy'
        Scoring metric to use.

    Returns
    -------
    scores : dict
        Dictionary containing the cross-validation scores.
    """
    import time

    from sklearn.model_selection import KFold

    # Validate input data
    X_checked, y_checked = check_data(X, y, ensure_2d=True)
    if y_checked is None:
        raise ValueError("Target values y must be provided for cross-validation")

    X = X_checked
    y = y_checked

    # Check if model is from our library or external (like sklearn)
    is_ml_library_model = hasattr(model, "train") and hasattr(model, "evaluate")

    # Initialize scores dictionary
    scores: Dict[str, List[float]] = {
        "test_score": [],
        "train_score": [],
        "fit_time": [],
        "score_time": [],
    }

    # Create KFold cross-validator
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Perform cross-validation
    for train_idx, test_idx in kfold.split(X):
        X_fold_train, X_fold_test = X[train_idx], X[test_idx]
        y_fold_train, y_fold_test = y[train_idx], y[test_idx]

        # Measure training time
        fit_start_time = time.time()
        if is_ml_library_model:
            model.train(X_fold_train, y_fold_train)
        else:
            model.fit(X_fold_train, y_fold_train)
        fit_end_time = time.time()
        fit_time = fit_end_time - fit_start_time

        # Measure scoring time and get test score
        score_start_time = time.time()
        if is_ml_library_model:
            fold_metrics = model.evaluate(X_fold_test, y_fold_test)
            train_metrics = model.evaluate(X_fold_train, y_fold_train)
        else:
            # For sklearn models, use score method
            test_score_val = model.score(X_fold_test, y_fold_test)
            train_score_val = model.score(X_fold_train, y_fold_train)
            fold_metrics = {scoring: test_score_val}
            train_metrics = {scoring: train_score_val}

        score_end_time = time.time()
        score_time = score_end_time - score_start_time

        # Extract the requested metric or default to first one
        if isinstance(fold_metrics, dict):
            if scoring in fold_metrics:
                test_score = fold_metrics[scoring]
            else:
                test_score = list(fold_metrics.values())[0]
        else:
            test_score = fold_metrics

        if isinstance(train_metrics, dict):
            if scoring in train_metrics:
                train_score = train_metrics[scoring]
            else:
                train_score = list(train_metrics.values())[0]
        else:
            train_score = train_metrics

        # Append scores to results
        scores["test_score"].append(test_score)
        scores["train_score"].append(train_score)
        scores["fit_time"].append(fit_time)
        scores["score_time"].append(score_time)

    # Convert lists to numpy arrays for consistency with sklearn's cross_validate
    scores_array = {}
    for key in scores:
        scores_array[key] = np.array(scores[key])

    return scores_array

    return scores
