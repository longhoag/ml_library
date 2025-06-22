"""Complete example of the ML library using a classification task."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer

from ml_library import (
    LogisticModel,
    RandomForestClassifier,
    StandardPreprocessor,
    accuracy,
    f1,
    plot_learning_curve,
    precision,
    recall,
    roc_auc,
    train_test_split,
)


def main():
    """Run a complete example using breast cancer dataset."""
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Preprocess the data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train and evaluate logistic regression model
    print("\n--- Logistic Regression Model ---")
    logistic_model = LogisticModel(C=1.0, max_iter=200)
    logistic_model.train(X_train_processed, y_train)

    # Make predictions
    y_pred_logistic = logistic_model.predict(X_test_processed)
    y_prob_logistic = logistic_model.predict_proba(X_test_processed)

    # Evaluate
    print(f"Accuracy: {accuracy(y_test, y_pred_logistic):.4f}")
    print(f"Precision: {precision(y_test, y_pred_logistic):.4f}")
    print(f"Recall: {recall(y_test, y_pred_logistic):.4f}")
    print(f"F1 Score: {f1(y_test, y_pred_logistic):.4f}")
    print(f"ROC AUC: {roc_auc(y_test, y_prob_logistic[:, 1]):.4f}")

    # Train and evaluate random forest model
    print("\n--- Random Forest Model ---")
    rf_model = RandomForestModel(n_estimators=100, random_state=42)
    rf_model.train(X_train_processed, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test_processed)
    y_prob_rf = rf_model.predict_proba(X_test_processed)

    # Evaluate
    print(f"Accuracy: {accuracy(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision(y_test, y_pred_rf):.4f}")
    print(f"Recall: {recall(y_test, y_pred_rf):.4f}")
    print(f"F1 Score: {f1(y_test, y_pred_rf):.4f}")
    print(f"ROC AUC: {roc_auc(y_test, y_prob_rf[:, 1]):.4f}")

    # Get feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top 10 features

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Top 10 Feature Importances")
    plt.barh(range(10), importances[indices], align="center")
    plt.yticks(range(10), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("feature_importances.png")

    # Plot learning curve for random forest
    plt.figure(figsize=(10, 6))
    lc_plot = plot_learning_curve(rf_model, X, y, cv=5)
    lc_plot.savefig("learning_curve.png")

    # Save the best model
    rf_model.save("breast_cancer_model.pkl")
    print("\nModel saved to breast_cancer_model.pkl")
    print("Feature importance plot saved to feature_importances.png")
    print("Learning curve plot saved to learning_curve.png")


if __name__ == "__main__":
    main()
