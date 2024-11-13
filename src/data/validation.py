"""Data validation module for ensuring data quality and consistency."""

from typing import Dict, List, Optional, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """Data validation class for checking data quality and consistency."""

    def __init__(self, config_path: str = "configs/feature_config.yaml"):
        """Initialize DataValidator with feature configuration.

        Args:
            config_path: Path to feature configuration file
        """
        self.config = load_config(config_path)
        self.numeric_features = self.config.get("numeric_features", [])
        self.categorical_features = self.config.get("categorical_features", [])
        self.derived_features = self.config.get("derived_features", [])

    def validate_schema(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate dataframe schema against configuration.

        Args:
            df: Input dataframe to validate

        Returns:
            Dictionary containing missing and extra columns
        """
        expected_columns = (
            self.numeric_features + self.categorical_features + self.derived_features
        )
        missing_columns = [col for col in expected_columns if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in expected_columns]

        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
        if extra_columns:
            logger.warning(f"Extra columns: {extra_columns}")

        return {"missing": missing_columns, "extra": extra_columns}

    def validate_data_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate data types of features.

        Args:
            df: Input dataframe to validate

        Returns:
            Dictionary containing type validation errors
        """
        type_errors = []

        # Validate numeric features
        for col in self.numeric_features:
            if col in df.columns and not is_numeric_dtype(df[col]):
                type_errors.append(f"{col} should be numeric")

        return {"type_errors": type_errors}

    def validate_value_ranges(
        self,
        df: pd.DataFrame,
        ranges: Optional[Dict[str, Dict[str, Union[float, int]]]] = None,
    ) -> Dict[str, List[str]]:
        """Validate value ranges for numeric features.

        Args:
            df: Input dataframe to validate
            ranges: Dictionary of valid ranges for features

        Returns:
            Dictionary containing range validation errors
        """
        range_errors = []

        if ranges is None:
            ranges = {
                "Age": {"min": 0, "max": 120},
                "Balance": {"min": 0, "max": float("inf")},
                "EstimatedSalary": {"min": 0, "max": float("inf")},
                "NumOfProducts": {"min": 1, "max": 10},
                "Tenure": {"min": 0, "max": 100},
            }

        for col, range_vals in ranges.items():
            if col in df.columns and is_numeric_dtype(df[col]):
                min_val, max_val = range_vals["min"], range_vals["max"]
                if df[col].min() < min_val or df[col].max() > max_val:
                    range_errors.append(
                        f"{col} values outside valid range [{min_val}, {max_val}]"
                    )

        return {"range_errors": range_errors}

    def validate_missing_values(
        self, df: pd.DataFrame, threshold: float = 0.1
    ) -> Dict[str, List[str]]:
        """Validate missing values in the dataset.

        Args:
            df: Input dataframe to validate
            threshold: Maximum allowed proportion of missing values

        Returns:
            Dictionary containing missing value validation results
        """
        missing_errors = []

        for col in df.columns:
            missing_prop = df[col].isna().mean()
            if missing_prop > threshold:
                missing_errors.append(
                    f"{col} has {missing_prop:.1%} missing values (threshold: {threshold:.1%})"
                )

        return {"missing_errors": missing_errors}

    def validate_categorical_values(
        self, df: pd.DataFrame, valid_categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[str]]:
        """Validate categorical values against allowed categories.

        Args:
            df: Input dataframe to validate
            valid_categories: Dictionary of valid categories for each feature

        Returns:
            Dictionary containing categorical validation errors
        """
        category_errors = []

        if valid_categories is None:
            valid_categories = {
                "Gender": ["Male", "Female"],
                "Geography": ["France", "Spain", "Germany"],
                "HasCrCard": [0, 1],
                "IsActiveMember": [0, 1],
            }

        for col, valid_cats in valid_categories.items():
            if col in df.columns:
                invalid_cats = set(df[col].unique()) - set(valid_cats)
                if invalid_cats:
                    category_errors.append(
                        f"{col} contains invalid categories: {invalid_cats}"
                    )

        return {"category_errors": category_errors}

    def validate_dataset(
        self,
        df: pd.DataFrame,
        ranges: Optional[Dict[str, Dict[str, Union[float, int]]]] = None,
        valid_categories: Optional[Dict[str, List[str]]] = None,
        missing_threshold: float = 0.1,
    ) -> Dict[str, List[str]]:
        """Perform comprehensive dataset validation.

        Args:
            df: Input dataframe to validate
            ranges: Dictionary of valid ranges for numeric features
            valid_categories: Dictionary of valid categories for categorical features
            missing_threshold: Maximum allowed proportion of missing values

        Returns:
            Dictionary containing all validation results
        """
        logger.info("Starting dataset validation")

        validation_results = {}

        # Schema validation
        validation_results.update(self.validate_schema(df))

        # Data type validation
        validation_results.update(self.validate_data_types(df))

        # Value range validation
        validation_results.update(self.validate_value_ranges(df, ranges))

        # Missing value validation
        validation_results.update(self.validate_missing_values(df, missing_threshold))

        # Categorical value validation
        validation_results.update(
            self.validate_categorical_values(df, valid_categories)
        )

        # Log validation results
        for check, errors in validation_results.items():
            if errors:
                logger.warning(f"{check}: {errors}")
            else:
                logger.info(f"{check}: Passed")

        return validation_results
