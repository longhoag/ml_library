"""Feature engineering utilities."""

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures

from ml_library.preprocessing import Preprocessor

__all__ = ["PolynomialPreprocessor", "FeatureSelector"]


class PolynomialPreprocessor(Preprocessor):
    """Preprocessor for generating polynomial features.

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.
    include_bias : bool, default=False
        If True, include a bias column.
    interaction_only : bool, default=False
        If True, only include interaction features.
    """

    def __init__(self, degree=2, include_bias=False, interaction_only=False):
        """Initialize the polynomial preprocessor."""
        super().__init__()
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.poly = PolynomialFeatures(
            degree=degree, include_bias=include_bias, interaction_only=interaction_only
        )

    def fit(self, X, y=None):
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target values.

        Returns
        -------
        self : PolynomialPreprocessor
            The fitted preprocessor.
        """
        self.poly.fit(X)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data with polynomial features.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        return self.poly.transform(X)


class FeatureSelector(Preprocessor):
    """Feature selection preprocessor.

    Parameters
    ----------
    k : int, default=10
        Number of top features to select.
    score_func : callable, default=None
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues). If None, f_classif is used for classification
        and f_regression for regression.
    """

    def __init__(self, k=10, score_func=None):
        """Initialize the feature selector."""
        super().__init__()
        self.k = k
        self.score_func = score_func
        self.selector = None

    def fit(self, X, y):
        """Fit the feature selector to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : FeatureSelector
            The fitted feature selector.
        """
        score_func = self.score_func

        # Auto-select score function if not provided
        if score_func is None:
            unique_y = np.unique(y)
            # If two unique values, assume classification
            if len(unique_y) <= 2 or (
                len(unique_y) <= 20 and np.all(np.mod(unique_y, 1) == 0)
            ):
                score_func = f_classif
            else:
                score_func = f_regression

        k = min(self.k, X.shape[1])  # Ensure k is not larger than n_features
        self.selector = SelectKBest(score_func, k=k)
        self.selector.fit(X, y)
        self.fitted = True
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Data with selected features.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        return self.selector.transform(X)

    @property
    def get_support(self):
        """Get a mask, or integer index, of the features selected.

        Returns
        -------
        mask : array
            Boolean mask of selected features.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before getting support mask")

        return self.selector.get_support()

    @property
    def scores_(self):
        """Get feature scores.

        Returns
        -------
        scores : array
            Feature scores.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before getting scores")

        return self.selector.scores_
