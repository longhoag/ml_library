"""Feature engineering transformations."""
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import PolynomialFeatures as SklearnPolynomialFeatures
from typing_extensions import Self

from ml_library.exceptions import PreprocessingError
from ml_library.preprocessing.base import Preprocessor


class PolynomialPreprocessor(Preprocessor):
    """Generate polynomial and interaction features.

    This transformer generates a new feature matrix consisting of all polynomial
    combinations of the features with degree less than or equal to the specified degree.
    """

    def __init__(
        self, degree: int = 2, interaction_only: bool = False, include_bias: bool = True
    ) -> None:
        """Initialize PolynomialPreprocessor.

        Args:
            degree: Maximum degree of the polynomial features.
            interaction_only: If true, only interaction features are produced.
            include_bias: If true, then include a bias column.
        """
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.transformer = SklearnPolynomialFeatures(
            degree=degree, interaction_only=interaction_only, include_bias=include_bias
        )

    def fit(self, X: NDArray[Any], y: Optional[NDArray[Any]] = None) -> Self:
        """Compute number of output features.

        Args:
            X: Training data.
            y: Target values (unused).

        Returns:
            self: Fitted instance.

        Raises:
            PreprocessingError: If input is not valid.
        """
        try:
            self.transformer.fit(X)
            self.fitted = True
            return self
        except Exception as e:
            raise PreprocessingError(
                f"Failed to fit polynomial features: {str(e)}"
            ) from e

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Transform data to polynomial features.

        Args:
            X: The data to transform.

        Returns:
            The polynomial features.

        Raises:
            PreprocessingError: If not fitted or transform fails.
        """
        if not self.fitted:
            raise PreprocessingError(
                "PolynomialPreprocessor must be fitted before transform"
            )
        try:
            result = self.transformer.transform(X)
            return np.asarray(result)
        except Exception as e:
            raise PreprocessingError(f"Failed to transform: {str(e)}") from e


class FeatureSelector(Preprocessor):
    """Select top k features based on variance."""

    def __init__(self, k: Optional[int] = None, feature_indices=None) -> None:
        """Initialize FeatureSelector.

        Args:
            k: Number of features to select. Required if feature_indices is None.
            feature_indices: Indices to select. If provided, k is ignored.

        Raises:
            ValueError: If k is less than 1 and feature_indices is None.
        """
        super().__init__()
        self.feature_indices = None

        if feature_indices is not None:
            # Convert feature_indices to numpy array if provided
            self.feature_indices = np.array(feature_indices, dtype=np.int_)
            self.k = len(feature_indices)
        else:
            if k is None or k < 1:
                raise ValueError("k must be at least 1 when feature_indices is None")
            self.k = k

        self._n_features = 0  # Initialize to 0 instead of None
        self.scores_ = None  # Add scores_ attribute

    def fit(self, X: NDArray[Any], y: Optional[NDArray[Any]] = None) -> Self:
        """Calculate feature variances and select top k.

        Args:
            X: Training data.
            y: Target values (unused).

        Returns:
            self: Fitted instance.

        Raises:
            PreprocessingError: If input is not valid or k is greater than
                number of features.
            ValueError: If input array is empty.
        """
        try:
            if X.size == 0:
                raise ValueError("Cannot fit with empty array")

            self._n_features = X.shape[1]

            # If feature_indices provided in __init__, use those
            if self.feature_indices is not None:
                # Validate feature indices
                if np.any(self.feature_indices >= self._n_features):
                    raise ValueError(
                        f"Feature index out of range. "
                        f"Max index is {self._n_features - 1}"
                    )
                self.fitted = True
                return self

            # Otherwise select based on variance
            if self.k > self._n_features:
                raise ValueError(
                    f"k ({self.k}) must not be greater than "
                    f"number of features ({self._n_features})"
                )

            # Calculate variance scores
            variances = np.var(X, axis=0)
            self.scores_ = variances  # Store variance scores
            self.feature_indices = np.argsort(variances)[-self.k :]
            self.fitted = True
            return self
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise PreprocessingError(f"Failed to fit feature selector: {str(e)}") from e

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Select top k features from input.

        Args:
            X: The data to transform.

        Returns:
            The selected features.

        Raises:
            PreprocessingError: If not fitted or transform fails.
            ValueError: If input shape does not match training shape.
        """
        if not self.fitted:
            raise PreprocessingError("FeatureSelector must be fitted before transform")
        if self.feature_indices is None:
            raise PreprocessingError("Feature indices not computed")

        try:
            if X.shape[1] != self._n_features:
                raise ValueError(
                    f"Shape of input is {X.shape}, but "
                    f"expected ({X.shape[0]}, {self._n_features})"
                )
            return X[:, self.feature_indices]
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise PreprocessingError(f"Failed to transform: {str(e)}") from e

    @property
    def get_support(self) -> NDArray[np.bool_]:
        """Get a boolean mask indicating selected features.

        Returns:
            Boolean array of shape [n_features], where True indicates a selected
            feature.

        Raises:
            PreprocessingError: If not fitted.
        """
        if not self.fitted:
            raise PreprocessingError(
                "FeatureSelector must be fitted before getting support mask"
            )
        if self.feature_indices is None:
            raise PreprocessingError("Feature indices not computed")

        mask = np.zeros(self._n_features, dtype=bool)
        mask[self.feature_indices] = True
        return mask
