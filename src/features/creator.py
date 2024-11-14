"""Feature creation module for the project."""

from typing import Optional

import pandas as pd

from src.utils.config import Config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureCreator:
    """Class for creating features from raw data."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize feature creator with configuration.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path) if config_path else Config()

        # Get required columns from config
        self.required_columns = (
            self.config.data.numeric_features + self.config.data.categorical_features
        )

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from input DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        logger.info("Starting feature creation")

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Check for encoded categorical columns
        for col in self.required_columns:
            if col not in df.columns:
                # Look for encoded versions of categorical columns
                encoded_cols = [c for c in df.columns if c.startswith(f"{col}_")]
                if not encoded_cols:
                    raise ValueError(
                        f"Missing required column '{col}' and no encoded version found"
                    )

        # Create derived features if specified
        if self.config.data.derived_features:
            for feature in self.config.data.derived_features:
                df = self._create_derived_feature(df, feature)

        logger.info(f"Created {len(df.columns)} total features")
        return df

    def _create_derived_feature(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """Create a single derived feature.

        Args:
            df: Input DataFrame
            feature: Name of feature to create

        Returns:
            DataFrame with new feature added
        """
        logger.info(f"Creating derived feature: {feature}")

        if feature == "balance_salary_ratio":
            df[feature] = df["Balance"] / df["EstimatedSalary"].replace(0, 1)

        elif feature == "products_per_tenure":
            df[feature] = df["NumOfProducts"] / df["Tenure"].replace(0, 1)

        elif feature == "active_with_credit_card":
            df[feature] = (df["IsActiveMember"] == 1) & (df["HasCrCard"] == 1)
            df[feature] = df[feature].astype(int)

        elif feature == "age_group":
            df[feature] = pd.cut(
                df["Age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=["18-25", "26-35", "36-45", "46-55", "55+"],
            )

        else:
            logger.warning(f"Unknown derived feature: {feature}")

        return df
