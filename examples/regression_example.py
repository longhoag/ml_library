"""Complete example of the ML library using a regression task."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes

from ml_library import (
    LinearModel,
    RandomForestRegressorModel,
    StandardPreprocessor,
    mae,
    mse,
    plot_learning_curve,
    r2,
    train_test_split,
)


def main():
    """Run a complete example using diabetes dataset."""
    # Load the dataset
    data = load_diabetes()
    X, y = data.data, data.target
    feature_names = data.feature_names

    print(f"Dataset shape: {X.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Preprocess the data
    preprocessor = StandardPreprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train and evaluate linear regression model
    print("\n--- Linear Regression Model ---")
    linear_model = LinearModel(fit_intercept=True)
    linear_model.train(X_train_processed, y_train)

    # Make predictions
    y_pred_linear = linear_model.predict(X_test_processed)

    # Evaluate
    print(f"Mean Squared Error: {mse(y_test, y_pred_linear):.2f}")
    print(f"Mean Absolute Error: {mae(y_test, y_pred_linear):.2f}")
    print(f"R² Score: {r2(y_test, y_pred_linear):.4f}")

    # Print coefficients
    print("\nModel Coefficients:")
    for name, coef in zip(feature_names, linear_model.coef_):
        print(f"{name}: {coef:.4f}")
    print(f"Intercept: {linear_model.intercept_:.4f}")

    # Train and evaluate random forest regressor model
    print("\n--- Random Forest Regressor Model ---")
    rf_model = RandomForestRegressorModel(n_estimators=100, random_state=42)
    rf_model.train(X_train_processed, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test_processed)

    # Evaluate
    print(f"Mean Squared Error: {mse(y_test, y_pred_rf):.2f}")
    print(f"Mean Absolute Error: {mae(y_test, y_pred_rf):.2f}")
    print(f"R² Score: {r2(y_test, y_pred_rf):.4f}")

    # Get feature importances
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-10:]  # Top features

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig("regression_feature_importances.png")

    # Plot predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_rf, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Predicted vs Actual Values")
    plt.tight_layout()
    plt.savefig("regression_predictions.png")

    # Plot learning curve
    plt.figure(figsize=(10, 6))
    lc_plot = plot_learning_curve(rf_model, X, y, cv=5)
    lc_plot.savefig("regression_learning_curve.png")

    # Save the best model
    rf_model.save("diabetes_model.pkl")
    print("\nModel saved to diabetes_model.pkl")
    print("Feature importance plot saved to regression_feature_importances.png")
    print("Predictions plot saved to regression_predictions.png")
    print("Learning curve plot saved to regression_learning_curve.png")


if __name__ == "__main__":
    main()
