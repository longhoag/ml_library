"""Data preprocessing utilities."""

__all__ = [
    "Preprocessor",
    "StandardPreprocessor",
    "PolynomialPreprocessor",
    "FeatureSelector",
]


class Preprocessor:
    """Base class for all data preprocessors."""

    def __init__(self):
        """Initialize the preprocessor."""
        self.fitted = False

    def fit(self, X, y=None):
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target values.

        Returns
        -------
        self : Preprocessor
            The fitted preprocessor.
        """
        self.fitted = True
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        return X

    def fit_transform(self, X, y=None):
        """Fit and transform the data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target values.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """
        return self.fit(X, y).transform(X)


from ml_library.preprocessing.feature_engineering import (
    FeatureSelector,
    PolynomialPreprocessor,
)

# Import specific preprocessors
from ml_library.preprocessing.standard import StandardPreprocessor
