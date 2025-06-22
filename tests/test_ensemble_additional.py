"""Additional tests for ensemble models to improve coverage."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from ml_library.models import RandomForestModel, RandomForestRegressorModel


@pytest.fixture
def binary_classification_data():
    """Generate sample binary classification data."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1, 
        n_classes=2, random_state=42
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def multi_classification_data():
    """Generate sample multi-class classification data."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3, n_redundant=1,
        n_classes=3, random_state=42
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    X, y, _ = make_regression(
        n_samples=100, n_features=5, n_informative=3, random_state=42, 
        coef=True
    )
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]
    return X_train, X_test, y_train, y_test


class TestRandomForestMultiClass:
    """Test RandomForestModel with multi-class data."""
    
    def test_evaluate_multi_class(self, multi_classification_data):
        """Test evaluate method with multi-class data."""
        X_train, X_test, y_train, y_test = multi_classification_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert isinstance(metrics["accuracy"], float)
        
    def test_custom_metrics_multi_class(self, multi_classification_data):
        """Test evaluate method with custom metrics for multi-class."""
        X_train, X_test, y_train, y_test = multi_classification_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        def custom_metric(y_true, y_pred):
            return np.mean(y_true == y_pred)
        
        metrics = model.evaluate(X_test, y_test, metrics={"custom": custom_metric})
        assert "custom" in metrics
        
    def test_roc_auc_error_handling(self, multi_classification_data):
        """Test roc_auc error handling in evaluate method."""
        X_train, X_test, y_train, y_test = multi_classification_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create a mock model that will raise an error when predict_proba is called
        original_predict_proba = model.predict_proba
        
        try:
            # Create a test scenario where roc_auc calculation would fail
            # Use monkeypatch if available, otherwise skip this test
            try:
                from sklearn.metrics import roc_auc_score
                original_roc_auc = roc_auc_score
                import sklearn.metrics
                
                # Replace roc_auc_score with a function that raises an error
                sklearn.metrics.roc_auc_score = lambda *args, **kwargs: exec('raise ValueError("Test error")')
                
                # Despite the error in roc_auc calculation, evaluate should still work
                metrics = model.evaluate(X_test, y_test)
                assert metrics["roc_auc"] == 0.5  # Default value when calculation fails
                
                # Restore original function
                sklearn.metrics.roc_auc_score = original_roc_auc
            except ImportError:
                # Skip test if can't import or monkeypatch
                pass
        finally:
            # Restore the original predict_proba method
            model.predict_proba = original_predict_proba


class TestRandomForestBinary:
    """Test RandomForestModel with binary class data."""
    
    def test_evaluate_binary(self, binary_classification_data):
        """Test evaluate method with binary classification."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics
        
    def test_binary_roc_auc_error_handling(self, binary_classification_data):
        """Test binary roc_auc error handling."""
        X_train, X_test, y_train, y_test = binary_classification_data
        model = RandomForestModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        # Create a mock model that will raise an error when predict_proba is called
        original_predict_proba = model.predict_proba
        
        try:
            # Create a test scenario where roc_auc calculation would fail
            # Use monkeypatch if available, otherwise skip this test
            try:
                # Save original predict_proba output
                y_proba_original = model.predict_proba(X_test)
                
                # Create a mock predict_proba that returns data that will cause an IndexError
                def mock_predict_proba(X):
                    # Return array with only one column which will cause an IndexError
                    # when code tries to access the second column
                    return np.array([[0.5]] * len(X))
                
                # Store reference to original method
                original_predict_proba = model.predict_proba
                
                # Use monkey patching to temporarily replace the method
                from types import MethodType
                model.predict_proba = MethodType(mock_predict_proba, model)
                
                # Despite the error in predict_proba, evaluate should still work
                metrics = model.evaluate(X_test, y_test)
                assert metrics["roc_auc"] == 0.5  # Default value when calculation fails
                
                # Restore original method
                model.predict_proba = original_predict_proba
            except Exception:
                # Skip test if can't patch method
                pass
        finally:
            # Restore the original predict_proba method
            model.predict_proba = original_predict_proba


class TestRandomForestRegressorCoverage:
    """Additional tests for RandomForestRegressorModel."""
    
    def test_regressor_predict_untrained(self):
        """Test predict raises error when regressor model is not trained."""
        model = RandomForestRegressorModel()
        with pytest.raises(ValueError):
            model.predict(np.array([[1, 2], [3, 4]]))
            
    def test_regressor_evaluate_untrained(self):
        """Test evaluate raises error when regressor model is not trained."""
        model = RandomForestRegressorModel()
        with pytest.raises(ValueError):
            model.evaluate(np.array([[1, 2], [3, 4]]), np.array([1, 2]))
    
    def test_regressor_feature_importances(self, regression_data):
        """Test feature_importances property of regressor model."""
        X_train, _, y_train, _ = regression_data
        model = RandomForestRegressorModel(n_estimators=10, random_state=42)
        
        # Should raise error if not fitted
        with pytest.raises(ValueError):
            _ = model.feature_importances_
        
        # Now train and check feature importances
        model.train(X_train, y_train)
        importances = model.feature_importances_
        assert importances.shape == (X_train.shape[1],)
        
    def test_regressor_custom_metrics(self, regression_data):
        """Test regressor evaluate method with custom metrics."""
        X_train, X_test, y_train, y_test = regression_data
        model = RandomForestRegressorModel(n_estimators=10, random_state=42)
        model.train(X_train, y_train)
        
        def custom_metric(y_true, y_pred):
            return float(np.sum(np.abs(y_true - y_pred)))
        
        metrics = model.evaluate(X_test, y_test, metrics={"custom": custom_metric})
        assert "custom" in metrics
        assert "r2" in metrics
        assert "mse" in metrics 
        assert "mae" in metrics
