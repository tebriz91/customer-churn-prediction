from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EDAVisualizer:
    """Class for creating exploratory data analysis visualizations."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize EDA visualizer with configuration.

        Args:
            config: Optional configuration object
        """
        self.config = config or Config()
        self.figure_size = (10, 6)
        plt.style.use(self.config.visualization.style)

    def plot_raw_data_issues(self, df: pd.DataFrame) -> plt.Figure:
        """Plot missing values and other data quality issues.

        Args:
            df: Input DataFrame to analyze

        Returns:
            matplotlib Figure object
        """
        logger.info("Generating missing value analysis plot")
        plt.figure(figsize=self.figure_size)
        sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis")
        plt.title("Missing Value Analysis")
        plt.tight_layout()
        return plt.gcf()

    def plot_numeric_distributions(
        self, df: pd.DataFrame, columns: Optional[List[str]] = None
    ) -> plt.Figure:
        """Plot distributions of numeric features.

        Args:
            df: Input DataFrame
            columns: Optional list of columns to plot. If None, plots all numeric columns.

        Returns:
            matplotlib Figure object
        """
        logger.info("Generating numeric feature distributions")

        # Select numeric columns if not specified
        if columns is None:
            columns = df.select_dtypes(include=["int64", "float64"]).columns

        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 plots per row

        fig = plt.figure(figsize=(15, 5 * n_rows))
        for i, column in enumerate(columns, 1):
            plt.subplot(n_rows, 3, i)
            sns.histplot(data=df, x=column, kde=True)
            plt.title(f"Distribution of {column}")
            plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def plot_feature_correlations(
        self, df: pd.DataFrame, target_col: Optional[str] = None
    ) -> plt.Figure:
        """Plot correlation matrix of features.

        Args:
            df: Input DataFrame
            target_col: Optional target column to highlight correlations with

        Returns:
            matplotlib Figure object
        """
        logger.info("Generating feature correlation matrix")

        # Calculate correlations
        corr_matrix = df.corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        plt.figure(figsize=self.figure_size)
        sns.heatmap(
            corr_matrix, mask=mask, annot=True, cmap="coolwarm", center=0, fmt=".2f"
        )
        plt.title("Feature Correlations")
        plt.tight_layout()
        return plt.gcf()

    def save_plots(
        self, df: pd.DataFrame, output_dir: str = "reports/figures/eda"
    ) -> None:
        """Generate and save all EDA plots.

        Args:
            df: Input DataFrame
            output_dir: Directory to save plots
        """
        try:
            # Create output directory if it doesn't exist
            self.config.paths.output_dir.mkdir(parents=True, exist_ok=True)

            # Generate and save plots
            plots = {
                "missing_values.png": self.plot_raw_data_issues(df),
                "numeric_distributions.png": self.plot_numeric_distributions(df),
                "feature_correlations.png": self.plot_feature_correlations(df),
            }

            # Save each plot
            for filename, fig in plots.items():
                output_path = f"{output_dir}/{filename}"
                fig.savefig(output_path)
                plt.close(fig)
                logger.info(f"Saved plot to {output_path}")

        except Exception as e:
            logger.error(f"Error saving plots: {str(e)}")
            raise


# # Usage example
# from src.visualization.eda import EDAVisualizer

# # Create visualizer
# eda = EDAVisualizer()

# # Generate individual plots
# missing_plot = eda.plot_raw_data_issues(df)
# dist_plot = eda.plot_numeric_distributions(df)
# corr_plot = eda.plot_feature_correlations(df)

# # Or generate and save all plots at once
# eda.save_plots(df)
