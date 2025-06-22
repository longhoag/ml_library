Basic Classification
==================

This tutorial walks through the process of setting up and training a classification model
using ML Library.

Loading and Preparing Data
------------------------

First, let's load some data and prepare it for training:

.. code-block:: python

    import numpy as np
    from sklearn.datasets import load_breast_cancer
    from ml_library.utils import train_test_split
    from ml_library.preprocessing import StandardPreprocessor

    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

Training a Model
--------------

Now let's train a logistic regression model:

.. code-block:: python

    from ml_library.models import LogisticModel

    # Initialize and train the model
    model = LogisticModel()
    model.fit(X_train_processed, y_train)

Making Predictions
---------------

Let's make predictions on the test set:

.. code-block:: python

    # Make predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)  # Probability estimates

Evaluating the Model
-----------------

Next, we'll evaluate the model's performance:

.. code-block:: python

    from ml_library.metrics import accuracy, precision, recall, f1, roc_auc

    # Calculate various metrics
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1_score = f1(y_test, y_pred)
    auc_score = roc_auc(y_test, y_pred_proba[:, 1])  # Use probability of class 1

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")

Visualizing Results
----------------

Finally, let's visualize the model's learning curve:

.. code-block:: python

    import matplotlib.pyplot as plt
    from ml_library.visualization import plot_learning_curve

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plot_learning_curve(
        model, X_train_processed, y_train,
        title="Learning Curve for Logistic Regression",
        cv=5
    )
    plt.show()

Complete Example
--------------

Here's the complete code:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_breast_cancer

    from ml_library.utils import train_test_split
    from ml_library.preprocessing import StandardPreprocessor
    from ml_library.models import LogisticModel
    from ml_library.metrics import accuracy, precision, recall, f1, roc_auc
    from ml_library.visualization import plot_learning_curve

    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Initialize and train the model
    model = LogisticModel()
    model.fit(X_train_processed, y_train)

    # Make predictions
    y_pred = model.predict(X_test_processed)
    y_pred_proba = model.predict_proba(X_test_processed)

    # Calculate various metrics
    acc = accuracy(y_test, y_pred)
    prec = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1_score = f1(y_test, y_pred)
    auc_score = roc_auc(y_test, y_pred_proba[:, 1])

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plot_learning_curve(
        model, X_train_processed, y_train,
        title="Learning Curve for Logistic Regression",
        cv=5
    )
    plt.show()
