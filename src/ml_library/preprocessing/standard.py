"""Standard data preprocessing implementation."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_library.preprocessing import Preprocessor

__all__ = ["StandardPreprocessor"]


class StandardPreprocessor(Preprocessor):
    """Standard preprocessing pipeline for mixed data types.

    This preprocessor applies standard scaling to numerical features
    and one-hot encoding to categorical features.

    Parameters
    ----------
    numerical_features : list or None, default=None
        Indices of numerical features. If None, all features are assumed to be numerical.
    categorical_features : list or None, default=None
        Indices of categorical features. If None, no features are assumed to be categorical.
    """

    def __init__(self, numerical_features=None, categorical_features=None):
        """Initialize the standard preprocessor."""
        super().__init__()
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.transformer = None

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
        self : StandardPreprocessor
            The fitted preprocessor.
        """
        # Initialize features if not specified
        if self.numerical_features is None and self.categorical_features is None:
            self.numerical_features = list(range(X.shape[1]))
            self.categorical_features = []

        transformers = []

        # Add numerical pipeline if needed
        if self.numerical_features:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            transformers.append(("num", num_pipeline, self.numerical_features))

        # Add categorical pipeline if needed
        if self.categorical_features:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        # Create column transformer
        self.transformer = ColumnTransformer(
            transformers=transformers, remainder="passthrough"
        )

        # Fit the transformer
        self.transformer.fit(X)
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
            Transformed data.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")

        return self.transformer.transform(X)
