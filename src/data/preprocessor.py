import os
from typing import Tuple

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Class for preprocessing customer churn data."""

    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first",
        )
        self.encoded_feature_mapping = {}  # Store mapping of original to encoded columns

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
        """Preprocess the input data.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (processed DataFrame, fitted scaler)
        """
        logger.info("Starting preprocessing pipeline...")

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df)

        # Handle outliers
        df = self._handle_outliers(df)

        # Scale numeric features
        df = self._scale_numeric_features(df)

        # Encode categorical features
        df = self._encode_categorical_features(df)

        return df, self.scaler

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Log missing value information
        missing_info = df.isnull().sum()
        if missing_info.any():
            logger.info("Missing values found:")
            logger.info("\n%s", missing_info[missing_info > 0])

        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if col != "Churn":  # Don't modify target variable
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)

        return df

    def _scale_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numeric features using StandardScaler."""
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        # Don't scale target or ID columns
        cols_to_scale = [
            col for col in numeric_cols if col not in ["Churn", "CustomerID"]
        ]

        if cols_to_scale:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            logger.info("Scaled numeric features: %s", cols_to_scale)

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using OneHotEncoder."""
        categorical_cols = df.select_dtypes(include=["object"]).columns

        if len(categorical_cols) > 0:
            # Fit and transform the categorical columns
            encoded = self.encoder.fit_transform(df[categorical_cols])
            # Get feature names with original column as prefix
            feature_names = self.encoder.get_feature_names_out(categorical_cols)

            # Store mapping of original to encoded columns
            for col in categorical_cols:
                encoded_cols = [f for f in feature_names if f.startswith(f"{col}_")]
                self.encoded_feature_mapping[col] = encoded_cols

            # Create DataFrame with encoded values
            encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)

            # Drop original categorical columns and add encoded ones
            df = df.drop(columns=categorical_cols)
            df = pd.concat([df, encoded_df], axis=1)

            logger.info("Encoded categorical features: %s", list(categorical_cols))

        return df

    def get_encoded_feature_names(self, original_col: str) -> list:
        """Get encoded feature names for an original column."""
        return self.encoded_feature_mapping.get(original_col, [])

    def analyze_data(self, df: pd.DataFrame, output_dir: str = "data") -> pd.DataFrame:
        """Analyze data and generate visualizations."""
        logger.info("Starting detailed data analysis")
        os.makedirs(output_dir, exist_ok=True)

        # Basic statistics
        logger.info("\nBasic Statistics:")
        logger.info("\n%s", df.describe(include="all"))

        # Missing values analysis
        logger.info("\nMissing Values Analysis:")
        logger.info("\n%s", df.isnull().sum())

        # Data types
        logger.info("\nData Types:")
        logger.info("\n%s", df.dtypes)

        return df
