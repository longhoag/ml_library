"""ML Library - A production-ready machine learning library."""

__version__ = "0.1.0"

# Import exceptions
from ml_library.exceptions import (
    DataError,
    InvalidParameterError,
    MLLibraryError,
    ModelError,
    NotFittedError,
    PreprocessingError,
    ValidationError,
)
# Import logging utilities
from ml_library.logging import configure_logging, get_logger
# Import metrics
from ml_library.metrics import accuracy, f1, mae, mse, precision, r2, recall, roc_auc
# Import models
from ml_library.models import (
    LinearModel,
    LogisticModel,
    Model,
    RandomForestModel,
    RandomForestRegressorModel,
)
# Import preprocessing utilities
from ml_library.preprocessing import (
    FeatureSelector,
    PolynomialPreprocessor,
    Preprocessor,
    StandardPreprocessor,
)
# Import utility functions
from ml_library.utils import check_data, cross_validate, train_test_split
# Import visualization utilities
from ml_library.visualization import plot_learning_curve

__all__ = [
    # Models
    "Model",
    "LinearModel",
    "LogisticModel",
    "RandomForestModel",
    "RandomForestRegressorModel",
    # Preprocessing
    "Preprocessor",
    "StandardPreprocessor",
    "PolynomialPreprocessor",
    "FeatureSelector",
    # Utils
    "check_data",
    "train_test_split",
    "cross_validate",
    # Visualization
    "plot_learning_curve",
    # Metrics
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mse",
    "mae",
    "r2",
    # Logging
    "get_logger",
    "configure_logging",
    # Exceptions
    "MLLibraryError",
    "DataError",
    "ModelError",
    "NotFittedError",
    "InvalidParameterError",
    "PreprocessingError",
    "ValidationError",
]
