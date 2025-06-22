"""ML Library - A production-ready machine learning library."""

__version__ = "0.1.0"

# Import key components for easier access
from ml_library.models import Model
from ml_library.preprocessing import Preprocessor
from ml_library.utils import check_data
from ml_library.visualization import plot_learning_curve

__all__ = ["Model", "Preprocessor", "check_data", "plot_learning_curve"]
