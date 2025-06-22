"""Standard preprocessing transformations."""

from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml_library.exceptions import PreprocessingError
from ml_library.preprocessing.base import Preprocessor

__all__ = ["StandardPreprocessor", "StandardScaler", "MinMaxScaler"]


def _handle_zeros_in_scale(scale: NDArray) -> NDArray:
    """Detect near-zero variance features in the scale array.

    Args:
        scale: Original scale array.

    Returns:
        Modified scale array with 1.0 where the original array had 0.
    """
    # Create a copy to avoid modifying original array
    scale_copy = scale.copy()
    if isinstance(scale_copy, np.ndarray):
        scale_copy[scale_copy == 0] = 1.0
    return scale_copy


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self) -> None:
        """Initialize a new StandardScaler."""
        self.mean_: Optional[NDArray[Any]] = None
        self.scale_: Optional[NDArray[Any]] = None
        self.fitted = False

    def fit(
        self, X: NDArray[Any], y: Optional[NDArray[Any]] = None
    ) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling.

        Args:
            X: Training data to calculate mean and std from.
            y: Ignored. Present for API consistency.

        Returns:
            self: The fitted scaler.

        Raises:
            PreprocessingError: If input contains NaN or infinity.
        """
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PreprocessingError("Input contains NaN or infinity.")

        self.mean_ = np.mean(X, axis=0)
        scale = np.std(X, axis=0)
        self.scale_ = _handle_zeros_in_scale(scale)
        self.fitted = True
        return self

    def transform(self, X: NDArray) -> NDArray:
        """Scale features using precomputed mean and std.

        Args:
            X: Input data to transform.

        Returns:
            X_scaled: Transformed data.

        Raises:
            PreprocessingError: If scaler is not fitted or input contains NaN/infinity.
        """
        if not self.fitted:
            raise PreprocessingError("StandardScaler is not fitted. Call fit() first.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PreprocessingError("Input contains NaN or infinity.")

        if self.mean_ is None or self.scale_ is None:
            raise PreprocessingError("Scaler is not properly fitted.")

        result = (X - self.mean_) / self.scale_
        return np.asarray(result)

    def fit_transform(
        self, X: NDArray[Any], y: Optional[NDArray[Any]] = None
    ) -> NDArray[Any]:
        """Fit to data, then transform it.

        Args:
            X: Input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            X_scaled: Transformed data.
        """
        return self.fit(X).transform(X)


class StandardPreprocessor(Preprocessor):
    """Standard preprocessing pipeline for mixed data types.

    This preprocessor applies standard scaling to numerical features
    and one-hot encoding to categorical features.

    Parameters
    ----------
    numerical_features : list or None, default=None
        Indices of numerical features. If None, all features are numerical.
    categorical_features : list or None, default=None
        Indices of categorical features. If None, no features are assumed to be
        categorical.
    """

    def __init__(
        self,
        numerical_features: Optional[list] = None,
        categorical_features: Optional[list] = None,
    ) -> None:
        """Initialize the standard preprocessor."""
        super().__init__()
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler = StandardScaler()  # Add scaler attribute

        # Create pipelines for numerical and categorical features
        num_pipeline = None
        if numerical_features is not None:
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

        cat_pipeline = None
        if categorical_features is not None:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

        # Create transformers list
        transformers = []
        if num_pipeline is not None:
            transformers.append(("num", num_pipeline, self.numerical_features))
        if cat_pipeline is not None:
            transformers.append(("cat", cat_pipeline, self.categorical_features))

        if transformers:
            self.transformer = ColumnTransformer(
                transformers=transformers,
                remainder="passthrough" if not transformers else "drop",
            )
        else:
            # If no specific features are provided, use a simple StandardScaler
            self.transformer = Pipeline([("scaler", StandardScaler())])

    def fit(
        self, X: NDArray[Any], y: Optional[NDArray[Any]] = None
    ) -> "StandardPreprocessor":
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target values.

        Returns
        -------
        self : StandardPreprocessor
            The fitted preprocessor.

        Raises
        ------
        PreprocessingError
            If fitting fails.
        """
        try:
            self.transformer.fit(X)
            self.scaler.fit(X)  # Also fit the standalone scaler
            self.fitted = True
            return self
        except Exception as e:
            raise PreprocessingError(f"Failed to fit preprocessor: {str(e)}") from e

    def transform(self, X: NDArray) -> Any:
        """Transform the data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data. May be a dense or sparse matrix.

        Raises
        ------
        PreprocessingError
            If transformation fails or preprocessor is not fitted.
        """
        if not self.fitted:
            raise PreprocessingError("Preprocessor must be fitted before transform")

        try:
            result = self.transformer.transform(X)
            return result
        except Exception as e:
            raise PreprocessingError(f"Failed to transform data: {str(e)}") from e


class MinMaxScaler:
    """Scale features to a given range.

    The default range is [0, 1]. The transformation is given by:
    X_scaled = (X - X_min) / (X_max - X_min)
    """

    def __init__(self, feature_range: tuple = (0, 1)) -> None:
        """Initialize MinMaxScaler.

        Args:
            feature_range: Desired range of transformed data.
        """
        self.feature_range = feature_range
        self.min_: Optional[NDArray[Any]] = None
        self.scale_: Optional[NDArray[Any]] = None
        self.data_min_: Optional[NDArray[Any]] = None
        self.data_max_: Optional[NDArray[Any]] = None
        self.data_range_: Optional[NDArray[Any]] = None
        self.fitted = False

    def fit(self, X: NDArray[Any], y: Optional[NDArray[Any]] = None) -> "MinMaxScaler":
        """Compute min and max to be used for scaling.

        Args:
            X: Training data to calculate min and max from.
            y: Ignored. Present for API consistency.

        Returns:
            self: The fitted scaler.

        Raises:
            PreprocessingError: If input contains NaN or infinity.
        """
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PreprocessingError("Input contains NaN or infinity.")

        feature_range = self.feature_range
        data_min = np.min(X, axis=0)
        data_max = np.max(X, axis=0)

        self.data_min_ = data_min
        self.data_max_ = data_max
        data_range = data_max - data_min
        self.data_range_ = _handle_zeros_in_scale(data_range)

        # Initialize scale_ as numpy array before division
        self.scale_ = np.ones_like(data_range) * (feature_range[1] - feature_range[0])

        # Ensure data_range_ is not None before division
        if self.data_range_ is not None:
            self.scale_ = self.scale_ / self.data_range_

        self.min_ = feature_range[0] - data_min * self.scale_
        self.fitted = True

        return self

    def transform(self, X: NDArray) -> NDArray:
        """Scale features using precomputed min and max.

        Args:
            X: Input data to transform.

        Returns:
            X_scaled: Transformed data.

        Raises:
            PreprocessingError: If scaler is not fitted or input contains NaN/infinity.
        """
        if not self.fitted:
            raise PreprocessingError("MinMaxScaler is not fitted. Call fit() first.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise PreprocessingError("Input contains NaN or infinity.")

        if self.scale_ is None or self.min_ is None:
            raise PreprocessingError("Scaler is not properly fitted.")

        result = X * self.scale_ + self.min_
        return np.asarray(result)

    def fit_transform(
        self, X: NDArray[Any], y: Optional[NDArray[Any]] = None
    ) -> NDArray[Any]:
        """Fit to data, then transform it.

        Args:
            X: Input data to fit and transform.
            y: Ignored. Present for API consistency.

        Returns:
            X_scaled: Transformed data.
        """
        return self.fit(X).transform(X)
