# Logging and Error Handling

This module provides logging and error handling utilities for the ML Library.

## Logging

The logging module allows for consistent logging across the entire library with configurable log levels and output destinations.

### Key Features

- **Consistent Logging**: All modules use the same logging configuration
- **Configurable**: Set log level, output destination, and format
- **Hierarchical**: Uses Python's built-in logging hierarchy

### Usage

```python
from ml_library.logging import get_logger, configure_logging

# Configure logging (call once at the start of your application)
configure_logging(level="info", log_file="my_app.log")

# Get a logger for your module
logger = get_logger(__name__)

# Use the logger
logger.debug("This is a debug message")
logger.info("Processing data with shape: %s", data.shape)
logger.warning("Missing values detected")
logger.error("Failed to process data")
logger.critical("Application cannot continue")

# Log exceptions
try:
    # Some code that might fail
    process_data()
except Exception as e:
    logger.exception("Error processing data: %s", str(e))
    # The exception traceback will be included automatically
```

## Error Handling

The library provides a set of custom exceptions that help with proper error reporting and handling.

### Exception Hierarchy

- `MLLibraryError` - Base exception for all library errors
  - `DataError` - Errors related to data validation or processing
  - `ModelError` - Base class for model-related errors
    - `NotFittedError` - Model used before training
  - `InvalidParameterError` - Invalid parameters provided
  - `PreprocessingError` - Errors in data preprocessing
  - `ValidationError` - Data validation errors

### Example Usage

```python
from ml_library.exceptions import DataError, NotFittedError

def process_data(X, y=None):
    try:
        # Validate data
        if X.shape[0] < 10:
            raise DataError("Not enough samples", data_shape=X.shape)
            
        # Process data
        return processed_data
    except Exception as e:
        # Wrap other exceptions in our custom exception
        raise DataError(f"Error processing data: {str(e)}", data_shape=X.shape) from e

def predict(model, X):
    if not model.trained:
        raise NotFittedError("Model must be trained before prediction", model_type=model.__class__.__name__)
    
    return model.predict(X)
```

## Best Practices

1. Always use lazy string formatting with logging:
   ```python
   # Good
   logger.info("Processing %d samples", n_samples)
   
   # Bad
   logger.info(f"Processing {n_samples} samples")
   ```

2. Use the appropriate log level:
   - `DEBUG`: Detailed information for debugging
   - `INFO`: Confirmation that things are working as expected
   - `WARNING`: Something unexpected happened but the code can continue
   - `ERROR`: A more serious problem that prevented the function from completing
   - `CRITICAL`: A very serious error that might prevent the program from continuing

3. Include context in error messages:
   ```python
   raise DataError("Invalid dimensions", data_shape=(X.shape, y.shape))
   ```

4. Use exception chaining to preserve the original traceback:
   ```python
   try:
       # Some code
   except ValueError as e:
       raise DataError("Error in data processing") from e
   ```
