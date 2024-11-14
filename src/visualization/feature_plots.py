"""Feature visualization module."""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeaturePlotter:
    """Class for creating feature-related visualizations."""

    def __init__(self, config: Optional[Config] = None, figsize: tuple = (10, 6)):
        """Initialize FeaturePlotter.

        Args:
            config: Optional configuration object
            figsize: Default figure size for plots
        """
        self.config = config or Config()
        self.figsize = figsize
        plt.style.use(self.config.visualization.style)

    def plot_feature_importance(
        self,
        feature_importance: Union[pd.DataFrame, dict],
        title: str = "Feature Importance",
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature importance scores.

        Args:
            feature_importance: Feature importance scores
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            if isinstance(feature_importance, dict):
                df = pd.DataFrame(
                    list(feature_importance.items()), columns=["feature", "importance"]
                )
            else:
                df = feature_importance.copy()

            # Sort and get top N features
            df = df.sort_values("importance", ascending=True).tail(top_n)

            plt.figure(figsize=self.figsize)
            sns.barplot(data=df, y="feature", x="importance")

            plt.title(title)
            plt.xlabel("Importance Score")
            plt.ylabel("Feature")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved feature importance plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise

    def plot_feature_correlations(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        title: str = "Feature Correlations",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature correlation heatmap.

        Args:
            df: Input dataframe
            target_col: Target column name
            title: Plot title
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            # Calculate correlations
            corr = df.corr()

            # Mask upper triangle
            mask = np.triu(np.ones_like(corr, dtype=bool))

            plt.figure(figsize=self.figsize)
            sns.heatmap(
                corr,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
            )

            plt.title(title)

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved correlation plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting correlations: {str(e)}")
            raise

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        features: List[str],
        target_col: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot feature distributions.

        Args:
            df: Input dataframe
            features: List of features to plot
            target_col: Target column name for conditional plots
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            n_features = len(features)
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
            axes = axes.flatten()

            for i, feature in enumerate(features):
                if target_col and df[feature].dtype in ["int64", "float64"]:
                    sns.boxplot(data=df, x=target_col, y=feature, ax=axes[i])
                else:
                    sns.histplot(data=df, x=feature, ax=axes[i])
                axes[i].set_title(f"{feature} Distribution")

            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved distribution plots to {save_path}")

            return fig

        except Exception as e:
            logger.error(f"Error plotting distributions: {str(e)}")
            raise

    def plot_mutual_information(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        title: str = "Mutual Information Scores",
        top_n: int = 10,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot mutual information scores between features and target.

        Args:
            X: Feature matrix
            y: Target variable
            title: Plot title
            top_n: Number of top features to show
            save_path: Path to save the plot

        Returns:
            Matplotlib figure object
        """
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y)
            mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores})

            # Sort and get top N features
            mi_df = mi_df.sort_values("mi_score", ascending=True).tail(top_n)

            plt.figure(figsize=self.figsize)
            sns.barplot(data=mi_df, y="feature", x="mi_score")

            plt.title(title)
            plt.xlabel("Mutual Information Score")
            plt.ylabel("Feature")

            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                logger.info(f"Saved mutual information plot to {save_path}")

            return plt.gcf()

        except Exception as e:
            logger.error(f"Error plotting mutual information: {str(e)}")
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
