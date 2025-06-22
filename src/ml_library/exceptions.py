"""Custom exceptions for ML Library."""

from typing import Any, Dict, Optional, Sequence

__all__ = [
    "MLLibraryError",
    "DataError",
    "ModelError",
    "NotFittedError",
    "InvalidParameterError",
    "PreprocessingError",
    "ValidationError",
]


class MLLibraryError(Exception):
    """Base exception class for all ML Library exceptions."""

    def __init__(self, message: str = "An error occurred in ML Library"):
        """Initialize the exception.

        Parameters
        ----------
        message : str, default="An error occurred in ML Library"
            Error message explaining the issue.
        """
        self.message = message
        super().__init__(self.message)


class DataError(MLLibraryError):
    """Exception for errors in data handling or validation."""

    def __init__(
        self, message: str = "Invalid data", data_shape: Optional[tuple] = None
    ):
        """Initialize the data error.

        Parameters
        ----------
        message : str, default="Invalid data"
            Error message explaining the issue.
        data_shape : tuple, optional
            Shape of the data that caused the error.
        """
        self.data_shape = data_shape
        shape_str = f" (shape: {data_shape})" if data_shape else ""
        full_message = f"{message}{shape_str}"
        super().__init__(full_message)


class ModelError(MLLibraryError):
    """Exception for general model errors."""

    def __init__(
        self,
        message: str = "Model error",
        model_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the model error.

        Parameters
        ----------
        message : str, default="Model error"
            Error message explaining the issue.
        model_type : str, optional
            Type of the model that raised the error.
        details : dict, optional
            Additional details about the error.
        """
        self.model_type = model_type
        self.details = details or {}

        model_info = f" in {model_type}" if model_type else ""
        full_message = f"{message}{model_info}"
        if details:
            full_message += f" (details: {details})"

        super().__init__(full_message)


class NotFittedError(ModelError):
    """Exception raised when a model is used before being fitted."""

    def __init__(
        self, message: str = "Model is not fitted", model_type: Optional[str] = None
    ):
        """Initialize the not fitted error.

        Parameters
        ----------
        message : str, default="Model is not fitted"
            Error message explaining the issue.
        model_type : str, optional
            Type of the model that raised the error.
        """
        super().__init__(message=message, model_type=model_type)


class InvalidParameterError(MLLibraryError):
    """Exception for invalid parameters."""

    def __init__(
        self,
        param_name: str,
        param_value: Any,
        allowed_values: Optional[Sequence[Any]] = None,
        message: Optional[str] = None,
    ):
        """Initialize the invalid parameter error.

        Parameters
        ----------
        param_name : str
            Name of the parameter that is invalid.
        param_value : any
            Value of the parameter that is invalid.
        allowed_values : sequence, optional
            If provided, the allowed values for this parameter.
        message : str, optional
            Custom error message. If None, a default message is generated.
        """
        self.param_name = param_name
        self.param_value = param_value
        self.allowed_values = allowed_values

        if message is None:
            message = f"Invalid value for parameter '{param_name}': {param_value}"
            if allowed_values is not None:
                message += f", allowed values are: {allowed_values}"

        super().__init__(message)


class PreprocessingError(MLLibraryError):
    """Exception for errors during data preprocessing."""

    def __init__(
        self,
        message: str = "Preprocessing error",
        preprocessor_type: Optional[str] = None,
        data_shape: Optional[tuple] = None,
    ):
        """Initialize the preprocessing error.

        Parameters
        ----------
        message : str, default="Preprocessing error"
            Error message explaining the issue.
        preprocessor_type : str, optional
            Type of the preprocessor that raised the error.
        data_shape : tuple, optional
            Shape of the data that caused the error.
        """
        self.preprocessor_type = preprocessor_type
        self.data_shape = data_shape

        info = []
        if preprocessor_type:
            info.append(f"preprocessor: {preprocessor_type}")
        if data_shape:
            info.append(f"data shape: {data_shape}")

        extra_info = f" ({', '.join(info)})" if info else ""
        full_message = f"{message}{extra_info}"

        super().__init__(full_message)


class ValidationError(MLLibraryError):
    """Exception for validation errors."""

    def __init__(
        self,
        message: str = "Validation error",
        context: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the validation error.

        Parameters
        ----------
        message : str, default="Validation error"
            Error message explaining the issue.
        context : dict, optional
            Additional context information related to the validation error.
        """
        self.context = context or {}

        full_message = message
        if context:
            full_message += f" (context: {context})"

        super().__init__(full_message)
