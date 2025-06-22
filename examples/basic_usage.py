"""Example usage of the ML library."""

# No numpy import needed here
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from ml_library import Model, Preprocessor


def main() -> None:
    """Run a simple example of the ML library."""
    # Generate a synthetic dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10, random_state=42
    )

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Preprocess the data
    preprocessor = Preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train, y_train)
    X_test_processed = preprocessor.transform(X_test)

    # Train the model
    model = Model()
    model.train(X_train_processed, y_train)

    # Evaluate the model
    score = model.evaluate(X_test_processed, y_test)
    print(f"Model accuracy: {score: .4f}")

    # Save the model
    model.save("model.pkl")
    print("Model saved to model.pkl")

    # Load the model
    loaded_model = Model.load("model.pkl")
    loaded_score = loaded_model.evaluate(X_test_processed, y_test)
    print(f"Loaded model accuracy: {loaded_score: .4f}")


if __name__ == "__main__":
    main()
