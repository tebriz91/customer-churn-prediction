"""Model evaluation visualization module."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluationPlotter:
    """Class for creating model evaluation visualizations."""

    def __init__(self, figsize: tuple = (10, 6)):
        """Initialize ModelEvaluationPlotter.

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        plt.style.use("seaborn")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot confusion matrix as a heatmap.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            cm = confusion_matrix(y_true, y_pred)

            plt.figure(figsize=self.figsize)
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )

            plt.title(title)
            plt.xlabel("Predicted")
            plt.ylabel("True")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved confusion matrix plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            raise

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot ROC curve with AUC score.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=self.figsize)
            plt.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"ROC curve (AUC = {roc_auc:.2f})",
            )
            plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")

            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(title)
            plt.legend(loc="lower right")
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved ROC curve plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            raise

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Precision-Recall Curve",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot precision-recall curve.

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)

            plt.figure(figsize=self.figsize)
            plt.plot(recall, precision, color="darkorange", lw=2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(title)
            plt.grid(True)

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved precision-recall curve plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting precision-recall curve: {str(e)}")
            raise

    def plot_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        title: str = "Model Comparison",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot model comparison across different metrics.

        Args:
            metrics: Dictionary of model metrics
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            # Prepare data for plotting
            df = pd.DataFrame(metrics).T

            plt.figure(figsize=self.figsize)
            df.plot(kind="bar", ax=plt.gca())

            plt.title(title)
            plt.xlabel("Model")
            plt.ylabel("Score")
            plt.legend(title="Metrics", bbox_to_anchor=(1.05, 1))
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved model comparison plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting model comparison: {str(e)}")
            raise


# # Usage example
# from src.visualization.evaluation import ModelEvaluationPlotter
# from src.visualization.feature_plots import FeaturePlotter

# # Create evaluation plots
# eval_plotter = ModelEvaluationPlotter()
# eval_plotter.plot_confusion_matrix(y_true, y_pred)
# eval_plotter.plot_roc_curve(y_true, y_prob)

# # Create feature plots
# feature_plotter = FeaturePlotter()
# feature_plotter.plot_feature_importance(importance_scores)
# feature_plotter.plot_feature_correlations(df)
