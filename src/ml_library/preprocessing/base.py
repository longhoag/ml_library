"""Base preprocessor class."""

from typing import Any, Optional

from numpy.typing import NDArray
from typing_extensions import Self


class Preprocessor:
    """Base class for all data preprocessors."""

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        self.fitted = False

    def fit(self, X: NDArray[Any], y: Optional[NDArray[Any]] = None) -> Self:
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

        Notes
        -----
        This base implementation just marks the preprocessor as fitted.
        Subclasses should override this method to implement actual fitting logic.
        """
        # pylint: disable=unused-argument
        self.fitted = True
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        """Transform the data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data.

        Raises
        ------
        NotImplementedError
            This method should be implemented by subclasses.
        """
        raise NotImplementedError(
            f"transform() not implemented for {self.__class__.__name__}"
        )

    def fit_transform(
        self, X: NDArray[Any], y: Optional[NDArray[Any]] = None
    ) -> NDArray[Any]:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : array-like
            Data to transform.
        y : array-like, optional
            Target values.

        Returns
        -------
        X_transformed : array-like
            Transformed data.
        """
        return self.fit(X, y).transform(X)
