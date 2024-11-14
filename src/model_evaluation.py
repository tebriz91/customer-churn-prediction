import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class for evaluating machine learning models for churn prediction."""

    def __init__(self, model_path: str = "models/churn_model.pkl"):
        """Initialize ModelEvaluator with a trained model."""
        try:
            self.model = joblib.load(model_path)
            # Store feature names from the trained model
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = self.model.feature_names_in_
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None  # Allow initialization without model for testing
            self.feature_names = None

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency between training and test data."""
        if not hasattr(self, "feature_names"):
            logger.warning("No feature names found in model. Proceeding with caution.")
            return X

        missing_cols = set(self.feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(self.feature_names)

        if missing_cols:
            logger.warning(f"Adding missing columns: {missing_cols}")
            for col in missing_cols:
                X[col] = 0

        if extra_cols:
            logger.warning(f"Removing extra columns: {extra_cols}")
            X = X.drop(columns=extra_cols)

        # Ensure columns are in the same order as during training
        return X[self.feature_names]

    def evaluate_model(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Tuple[Dict[str, float], str]:
        """
        Evaluate model performance using multiple metrics.

        Args:
            X_test: Test features
            y_test: True labels

        Returns:
            Tuple containing metrics dictionary and classification report
        """
        try:
            # Align features with training data
            X_test_aligned = self._validate_features(X_test)

            predictions = self.model.predict(X_test_aligned)
            probabilities = self.model.predict_proba(X_test_aligned)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, predictions),
                "precision": precision_score(y_test, predictions),
                "recall": recall_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }

            report = classification_report(y_test, predictions)

            # Log the results
            logger.info("Model Evaluation Results:")
            for metric, value in metrics.items():
                logger.info(f"{metric.capitalize()}: {value:.3f}")

            return metrics, report

        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise

    def plot_confusion_matrix(
        self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "data"
    ) -> None:
        """Plot and save confusion matrix."""
        try:
            X_test_aligned = self._validate_features(X_test)
            predictions = self.model.predict(X_test_aligned)
            cm = confusion_matrix(y_test, predictions)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                annot_kws={"size": 14},
                square=True,
                cbar_kws={"label": "Count"},
            )
            plt.title("Confusion Matrix", fontsize=14, pad=20)
            plt.ylabel("True Label", fontsize=12)
            plt.xlabel("Predicted Label", fontsize=12)

            # Add metrics annotations with better formatting
            metrics = {
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions),
                "Recall": recall_score(y_test, predictions),
                "F1": f1_score(y_test, predictions),
            }

            metrics_text = "\n".join(f"{k}: {v:.2f}" for k, v in metrics.items())
            plt.text(
                2.3,
                1.2,
                metrics_text,
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                    boxstyle="round,pad=0.5",
                ),
            )

            # Add sample size annotation with better formatting
            plt.text(
                -0.2,
                -0.2,
                f"Test samples: {len(y_test)}",
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                    boxstyle="round,pad=0.2",
                ),
            )

            plt.tight_layout()
            output_path = Path(output_dir) / "confusion_matrix_eval.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Confusion matrix plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

    def plot_roc_curve(
        self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "data"
    ) -> None:
        """Plot and save ROC curve."""
        try:
            X_test_aligned = self._validate_features(X_test)
            probabilities = self.model.predict_proba(X_test_aligned)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probabilities)
            roc_auc = roc_auc_score(y_test, probabilities)

            plt.figure(figsize=(10, 8))
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot(
                [0, 1],
                [0, 1],
                color="navy",
                lw=2,
                linestyle="--",
                label="Random baseline",
            )
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curve", fontsize=14, pad=20)
            plt.legend(loc="lower right", fontsize=10)

            # Improved grid
            plt.grid(True, alpha=0.3, linestyle="--")

            # Add sample size annotation with better formatting
            plt.text(
                0.05,
                0.95,
                f"Test samples: {len(y_test)}",
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="gray",
                    boxstyle="round,pad=0.2",
                ),
            )

            plt.tight_layout()
            output_path = Path(output_dir) / "roc_curve_eval.png"
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"ROC curve plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise

    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Generate a classification report."""
        if self.model is None:
            raise ValueError("No model available for evaluation")
        return classification_report(y_true, y_pred)

    def analyze_errors(self, X: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
        """Analyze prediction errors."""
        if self.model is None:
            raise ValueError("No model available for error analysis")

        X_aligned = self._validate_features(X)
        predictions = self.model.predict(X_aligned)
        probabilities = self.model.predict_proba(X_aligned)[:, 1]

        error_analysis = X.copy()
        error_analysis["actual"] = y_true
        error_analysis["predicted"] = predictions
        error_analysis["probability"] = probabilities
        error_analysis["error"] = y_true != predictions
        error_analysis["confidence"] = np.where(
            probabilities > 0.5, probabilities, 1 - probabilities
        )

        return error_analysis[error_analysis["error"]]


def main():
    """Main function to run the model evaluation pipeline."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path("data")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load test data
        logger.info("Loading test data...")
        X_test = pd.read_csv("data/X_test_featured.csv")
        y_test = pd.read_csv("data/y_test.csv").values.ravel()

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Evaluate model
        metrics, report = evaluator.evaluate_model(X_test, y_test)
        logger.info(f"\nClassification Report:\n{report}")

        # Generate visualization plots
        evaluator.plot_confusion_matrix(X_test, y_test)
        evaluator.plot_roc_curve(X_test, y_test)

        # Analyze errors
        error_analysis = evaluator.analyze_errors(X_test, y_test)
        error_analysis_path = output_dir / "error_analysis.csv"
        error_analysis.to_csv(error_analysis_path, index=False)
        logger.info(f"Error analysis saved to {error_analysis_path}")

        logger.info("Model evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
