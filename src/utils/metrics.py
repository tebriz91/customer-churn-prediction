"""Metrics module for model evaluation."""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


def calculate_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
    average: str = "binary",
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        average: Averaging strategy for multiclass problems

    Returns:
        Dictionary containing calculated metrics
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=average),
            "recall": recall_score(y_true, y_pred, average=average),
            "f1": f1_score(y_true, y_pred, average=average),
        }

        # Add probability-based metrics if probabilities are provided
        if y_prob is not None:
            metrics.update({
                "roc_auc": roc_auc_score(y_true, y_prob),
                "avg_precision": average_precision_score(y_true, y_prob),
            })

        logger.debug("Calculated classification metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating classification metrics: {str(e)}")
        raise


def calculate_confusion_matrix_metrics(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]
) -> Dict[str, Union[float, int]]:
    """Calculate metrics from confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing confusion matrix metrics
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        }

        logger.debug("Calculated confusion matrix metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating confusion matrix metrics: {str(e)}")
        raise


def calculate_class_distribution(y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Calculate class distribution metrics.

    Args:
        y: Labels

    Returns:
        Dictionary containing class distribution metrics
    """
    try:
        total = len(y)
        class_counts = pd.Series(y).value_counts()

        distribution = {
            f"class_{label}_count": count for label, count in class_counts.items()
        }

        distribution.update({
            f"class_{label}_percentage": (count / total) * 100
            for label, count in class_counts.items()
        })

        logger.debug("Calculated class distribution metrics")
        return distribution

    except Exception as e:
        logger.error(f"Error calculating class distribution: {str(e)}")
        raise


def calculate_prediction_confidence(
    y_prob: Union[np.ndarray, pd.Series], bins: int = 10
) -> Dict[str, Union[float, Dict[str, int]]]:
    """Calculate prediction confidence metrics.

    Args:
        y_prob: Predicted probabilities
        bins: Number of confidence bins

    Returns:
        Dictionary containing confidence metrics
    """
    try:
        confidence_scores = np.max(y_prob, axis=1) if len(y_prob.shape) > 1 else y_prob

        metrics = {
            "mean_confidence": float(np.mean(confidence_scores)),
            "median_confidence": float(np.median(confidence_scores)),
            "min_confidence": float(np.min(confidence_scores)),
            "max_confidence": float(np.max(confidence_scores)),
        }

        # Calculate confidence distribution
        hist, bin_edges = np.histogram(confidence_scores, bins=bins, range=(0, 1))
        confidence_dist = {
            f"confidence_{i}_{i + 1}": int(count) for i, count in enumerate(hist)
        }

        metrics["confidence_distribution"] = confidence_dist

        logger.debug("Calculated prediction confidence metrics")
        return metrics

    except Exception as e:
        logger.error(f"Error calculating prediction confidence: {str(e)}")
        raise


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Generate comprehensive classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        labels: List of label names

    Returns:
        Dictionary containing all evaluation metrics
    """
    try:
        report = {}

        # Basic classification metrics
        report["classification_metrics"] = calculate_classification_metrics(
            y_true, y_pred, y_prob
        )

        # Confusion matrix metrics
        report["confusion_matrix_metrics"] = calculate_confusion_matrix_metrics(
            y_true, y_pred
        )

        # Class distribution
        report["class_distribution"] = calculate_class_distribution(y_true)

        # Confidence metrics if probabilities are provided
        if y_prob is not None:
            report["confidence_metrics"] = calculate_prediction_confidence(y_prob)

        # Detailed classification report
        report["detailed_classification"] = classification_report(
            y_true, y_pred, labels=labels, output_dict=True
        )

        logger.info("Generated comprehensive classification report")
        return report

    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        raise


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Union[np.ndarray, None] = None,
) -> Dict[str, float]:
    """Calculate classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional)

    Returns:
        Dictionary containing various classification metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }

    # Add ROC AUC if probabilities are provided
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return metrics


# # Usage example
# from src.utils.metrics import generate_classification_report

# # Generate comprehensive report
# report = generate_classification_report(y_true, y_pred, y_prob)

# # Calculate specific metrics
# class_metrics = calculate_classification_metrics(y_true, y_pred)
# conf_matrix_metrics = calculate_confusion_matrix_metrics(y_true, y_pred)
