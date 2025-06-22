Quickstart
==========

This guide will help you get started with ML Library in just a few minutes.

Classification Example
---------------------

Here's a simple classification example using ML Library:

.. code-block:: python

    from ml_library.models import LogisticModel
    from ml_library.preprocessing import StandardPreprocessor
    from ml_library.utils import train_test_split
    from ml_library.metrics import accuracy, precision, recall, f1
    import numpy as np

    # Generate some dummy data
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Binary target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Preprocess data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train a logistic regression model
    model = LogisticModel()
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Evaluate the model
    print(f"Accuracy: {accuracy(y_test, y_pred):.4f}")
    print(f"Precision: {precision(y_test, y_pred):.4f}")
    print(f"Recall: {recall(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1(y_test, y_pred):.4f}")

Regression Example
----------------

Here's a simple regression example:

.. code-block:: python

    from ml_library.models import LinearModel
    from ml_library.preprocessing import StandardPreprocessor
    from ml_library.utils import train_test_split
    from ml_library.metrics import mse, mae, r2
    import numpy as np

    # Generate some dummy data
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    y = 2 * X[:, 0] - 1 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(100) * 0.1  # Linear target with noise

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Preprocess data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train a linear regression model
    model = LinearModel()
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)

    # Evaluate the model
    print(f"MSE: {mse(y_test, y_pred):.4f}")
    print(f"MAE: {mae(y_test, y_pred):.4f}")
    print(f"R² Score: {r2(y_test, y_pred):.4f}")

Next Steps
---------

- For more detailed examples, see the :doc:`examples` section.
- For advanced usage, see the :doc:`tutorials/index` section.
- For API reference, see the :doc:`api/index` section.
