import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.model_evaluation import ModelEvaluator


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


def test_generate_classification_report(sample_predictions):
    """Test classification report generation"""
    y_true, y_pred, _ = sample_predictions
    evaluator = ModelEvaluator()

    report = evaluator.generate_report(y_true, y_pred)
    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert "f1-score" in report


def test_error_analysis(sample_predictions):
    """Test error analysis functionality"""
    y_true, y_pred, _ = sample_predictions

    # Create sample features
    X = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [0.1, 0.2, 0.3, 0.4, 0.5],
    })

    evaluator = ModelEvaluator()
    error_analysis = evaluator.analyze_errors(X, y_true, y_pred)

    assert isinstance(error_analysis, pd.DataFrame)
    assert "actual" in error_analysis.columns
    assert "predicted" in error_analysis.columns
    assert len(error_analysis) == sum(y_true != y_pred)
