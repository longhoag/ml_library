"""Example of using feature engineering techniques."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression

from ml_library import (
    FeatureSelector,
    LinearModel,
    PolynomialPreprocessor,
    StandardPreprocessor,
    mse,
    r2,
    train_test_split,
)


def main():
    """Run feature engineering example."""
    # Generate synthetic regression data with only 5 informative features
    X, y = make_regression(
        n_samples=200, n_features=20, n_informative=5, noise=20, random_state=42
    )

    print(f"Dataset shape: {X.shape}")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # ----------------------------------------------------------
    # Approach 1: Standard preprocessing only
    # ----------------------------------------------------------
    print("\n--- Approach 1: Standard preprocessing only ---")

    # Apply standard preprocessing
    std_preprocessor = StandardPreprocessor()
    X_train_std = std_preprocessor.fit_transform(X_train)
    X_test_std = std_preprocessor.transform(X_test)

    # Train linear model
    model_std = LinearModel()
    model_std.train(X_train_std, y_train)

    # Evaluate
    y_pred_std = model_std.predict(X_test_std)
    r2_std = r2(y_test, y_pred_std)
    mse_std = mse(y_test, y_pred_std)

    print(f"R² Score: {r2_std:.4f}")
    print(f"MSE: {mse_std:.4f}")

    # ----------------------------------------------------------
    # Approach 2: Feature selection
    # ----------------------------------------------------------
    print("\n--- Approach 2: Feature selection ---")

    # Apply standard preprocessing then feature selection
    std_preprocessor = StandardPreprocessor()
    X_train_std = std_preprocessor.fit_transform(X_train)
    X_test_std = std_preprocessor.transform(X_test)

    # Select top features
    selector = FeatureSelector(k=5)
    X_train_selected = selector.fit_transform(X_train_std, y_train)
    X_test_selected = selector.transform(X_test_std)

    # Train linear model on selected features
    model_selected = LinearModel()
    model_selected.train(X_train_selected, y_train)

    # Evaluate
    y_pred_selected = model_selected.predict(X_test_selected)
    r2_selected = r2(y_test, y_pred_selected)
    mse_selected = mse(y_test, y_pred_selected)

    print(f"R² Score: {r2_selected:.4f}")
    print(f"MSE: {mse_selected:.4f}")

    # Plot feature importance scores
    feature_scores = selector.scores_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_scores)), feature_scores)
    plt.xlabel("Feature Index")
    plt.ylabel("Score")
    plt.title("Feature Importance Scores")
    plt.tight_layout()
    plt.savefig("feature_selection_scores.png")

    # ----------------------------------------------------------
    # Approach 3: Polynomial features
    # ----------------------------------------------------------
    print("\n--- Approach 3: Polynomial features ---")

    # First select top 3 features to avoid dimensionality explosion
    selector_first = FeatureSelector(k=3)
    X_train_selected_first = selector_first.fit_transform(X_train, y_train)
    X_test_selected_first = selector_first.transform(X_test)

    # Then apply polynomial features
    poly_preprocessor = PolynomialPreprocessor(degree=2)
    X_train_poly = poly_preprocessor.fit_transform(X_train_selected_first)
    X_test_poly = poly_preprocessor.transform(X_test_selected_first)

    # Train linear model on polynomial features
    model_poly = LinearModel()
    model_poly.train(X_train_poly, y_train)

    # Evaluate
    y_pred_poly = model_poly.predict(X_test_poly)
    r2_poly = r2(y_test, y_pred_poly)
    mse_poly = mse(y_test, y_pred_poly)

    print(f"R² Score: {r2_poly:.4f}")
    print(f"MSE: {mse_poly:.4f}")

    # Compare results
    print("\n--- Comparison ---")
    results = {
        "Standard": {"R²": r2_std, "MSE": mse_std},
        "Feature Selection": {"R²": r2_selected, "MSE": mse_selected},
        "Polynomial": {"R²": r2_poly, "MSE": mse_poly},
    }

    for name, metrics in results.items():
        print(f"{name}: R² = {metrics['R²']:.4f}, MSE = {metrics['MSE']:.4f}")

    # Plot comparison of predictions
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.scatter(y_test, y_pred_std)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title(f"Standard - R²: {r2_std:.4f}")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    plt.subplot(222)
    plt.scatter(y_test, y_pred_selected)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title(f"Feature Selection - R²: {r2_selected:.4f}")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    plt.subplot(223)
    plt.scatter(y_test, y_pred_poly)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.title(f"Polynomial - R²: {r2_poly:.4f}")
    plt.xlabel("True Values")
    plt.ylabel("Predictions")

    plt.tight_layout()
    plt.savefig("feature_engineering_comparison.png")
    print("Comparison plot saved to feature_engineering_comparison.png")
    print("Feature selection scores saved to feature_selection_scores.png")


if __name__ == "__main__":
    main()
