"""Feature transformation module for the project."""

from typing import Dict, List, Optional, Union

import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureTransformer:
    """Feature transformation class implementing various transformation methods."""

    def __init__(
        self,
        config_path: str = "configs/feature_config.yaml",
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
    ):
        """Initialize FeatureTransformer.

        Args:
            config_path: Path to feature configuration file
            scaling_method: Method for scaling numeric features
            encoding_method: Method for encoding categorical features
        """
        self.config = load_config(config_path)
        self.scaling_method = scaling_method
        self.encoding_method = encoding_method

        self.numeric_features = self.config.get("numeric_features", [])
        self.categorical_features = self.config.get("categorical_features", [])

        # Initialize transformers
        self.numeric_transformer = self._get_numeric_transformer(scaling_method)
        self.categorical_transformer = self._get_categorical_transformer(
            encoding_method
        )

        # Store fitted attributes
        self.categorical_mappings: Dict[str, Union[LabelEncoder, OneHotEncoder]] = {}
        self.feature_names_: Optional[List[str]] = None

    def _get_numeric_transformer(
        self, method: str
    ) -> Union[StandardScaler, MinMaxScaler]:
        """Get numeric feature transformer.

        Args:
            method: Scaling method ('standard' or 'minmax')

        Returns:
            Scaler object
        """
        if method == "standard":
            return StandardScaler()
        elif method == "minmax":
            return MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

    def _get_categorical_transformer(
        self, method: str
    ) -> Union[LabelEncoder, OneHotEncoder]:
        """Get categorical feature transformer.

        Args:
            method: Encoding method ('label' or 'onehot')

        Returns:
            Encoder object
        """
        if method == "label":
            return LabelEncoder()
        elif method == "onehot":
            return OneHotEncoder(sparse=False, handle_unknown="ignore")
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    def fit(self, X: pd.DataFrame) -> "FeatureTransformer":
        """Fit the feature transformer.

        Args:
            X: Feature matrix

        Returns:
            self: Fitted transformer
        """
        # Fit numeric features
        if self.numeric_features:
            self.numeric_transformer.fit(X[self.numeric_features])
            logger.info(
                f"Fitted numeric transformer for {len(self.numeric_features)} features"
            )

        # Fit categorical features
        if self.categorical_features:
            if self.encoding_method == "label":
                for col in self.categorical_features:
                    encoder = LabelEncoder()
                    encoder.fit(X[col].astype(str))
                    self.categorical_mappings[col] = encoder
            else:  # onehot
                self.categorical_transformer.fit(X[self.categorical_features])
                self.categorical_mappings = {"onehot": self.categorical_transformer}

            logger.info(
                f"Fitted categorical transformer for {len(self.categorical_features)} features"
            )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        result_parts = []

        # Transform numeric features
        if self.numeric_features:
            numeric_transformed = pd.DataFrame(
                self.numeric_transformer.transform(X[self.numeric_features]),
                columns=self.numeric_features,
                index=X.index,
            )
            result_parts.append(numeric_transformed)

        # Transform categorical features
        if self.categorical_features:
            if self.encoding_method == "label":
                categorical_transformed = pd.DataFrame(index=X.index)
                for col in self.categorical_features:
                    encoder = self.categorical_mappings[col]
                    categorical_transformed[col] = encoder.transform(X[col].astype(str))
            else:  # onehot
                encoder = self.categorical_mappings["onehot"]
                feature_names = []
                for i, col in enumerate(self.categorical_features):
                    feature_names.extend([
                        f"{col}_{val}" for val in encoder.categories_[i]
                    ])
                categorical_transformed = pd.DataFrame(
                    encoder.transform(X[self.categorical_features]),
                    columns=feature_names,
                    index=X.index,
                )
            result_parts.append(categorical_transformed)

        # Combine all parts
        result = pd.concat(result_parts, axis=1)
        self.feature_names_ = list(result.columns)

        return result

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        return self.fit(X).transform(X)

    def get_feature_names(self) -> List[str]:
        """Get names of transformed features.

        Returns:
            List of feature names
        """
        if self.feature_names_ is None:
            raise ValueError("Transformer has not been fitted yet")
        return self.feature_names_
