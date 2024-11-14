import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.model_evaluation import ModelEvaluator


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    return model


@pytest.fixture
def sample_predictions():
    """Create sample predictions for testing"""
    y_true = np.array([0, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.3, 0.4, 0.7])
    return y_true, y_pred, y_prob


def test_calculate_metrics(sample_predictions):
    """Test metric calculation"""
    y_true, y_pred, _ = sample_predictions

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Verify metrics are in valid range
    assert 0 <= accuracy <= 1
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= f1 <= 1


def test_generate_classification_report(sample_predictions, monkeypatch, sample_model):
    """Test classification report generation"""
    y_true, y_pred, _ = sample_predictions

    # Create evaluator with mocked model
    evaluator = ModelEvaluator()
    evaluator.model = sample_model  # Replace the model directly

    report = evaluator.generate_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report


def test_error_analysis(sample_predictions, sample_model):
    """Test error analysis functionality"""
    y_true, y_pred, _ = sample_predictions

    # Create sample features
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
    })

    # Create evaluator with mocked model
    evaluator = ModelEvaluator()
    evaluator.model = sample_model  # Replace the model directly
    evaluator.feature_names = X.columns  # Set feature names to match test data

    error_analysis = evaluator.analyze_errors(X, y_true)

    assert isinstance(error_analysis, pd.DataFrame)
    assert "actual" in error_analysis.columns
    assert "predicted" in error_analysis.columns
    assert len(error_analysis) == sum(y_true != evaluator.model.predict(X))


def test_model_evaluator_initialization(sample_model, tmp_path):
    """Test ModelEvaluator initialization with a saved model"""
    import joblib

    # Save the sample model to a temporary file
    model_path = tmp_path / "test_model.pkl"
    joblib.dump(sample_model, model_path)

    # Initialize evaluator with the saved model
    evaluator = ModelEvaluator(model_path=str(model_path))

    assert evaluator.model is not None
    assert hasattr(evaluator.model, "predict")
    assert hasattr(evaluator.model, "predict_proba")
