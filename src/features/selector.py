"""Feature selection module for the project."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Feature selection class implementing various selection methods."""

    def __init__(
        self,
        config_path: str = "configs/feature_config.yaml",
        k: int = 10,
        random_state: int = 42,
    ):
        """Initialize FeatureSelector."""
        self.config = load_config(config_path)
        self.k = k
        self.random_state = random_state
        self.selected_features: Optional[List[str]] = None
        self.feature_scores_: Optional[Dict[str, float]] = None

        # Initialize different selectors
        self.univariate_selector = SelectKBest(score_func=f_classif, k=k)
        self.mutual_info_selector = SelectKBest(score_func=mutual_info_classif, k=k)
        self.importance_selector = RandomForestClassifier(
            n_estimators=100, random_state=random_state
        )

    def fit(
        self, X: pd.DataFrame, y: pd.Series, method: str = "univariate"
    ) -> "FeatureSelector":
        """Fit the feature selector.

        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('univariate', 'mutual_info',
                   'importance', 'correlation')

        Returns:
            self: Fitted selector
        """
        if method == "univariate":
            self._fit_univariate(X, y)
        elif method == "mutual_info":
            self._fit_mutual_info(X, y)
        elif method == "importance":
            self._fit_importance(X, y)
        elif method == "correlation":
            self._fit_correlation(X, y)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        return self

    def _fit_importance(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit importance-based feature selector."""
        self.importance_selector.fit(X, y)
        importance_scores = self.importance_selector.feature_importances_

        # Get top k features
        top_indices = np.argsort(importance_scores)[-self.k :]
        self.selected_features = list(X.columns[top_indices])
        self.feature_scores_ = dict(zip(X.columns, importance_scores))

        # Log selection results
        self._log_selection_results()

    def _fit_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit mutual information-based feature selector."""
        self.mutual_info_selector.fit(X, y)
        mask = self.mutual_info_selector.get_support()
        self.selected_features = list(X.columns[mask])
        self.feature_scores_ = dict(zip(X.columns, self.mutual_info_selector.scores_))

        # Log selection results
        self._log_selection_results()

    def _fit_univariate(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit univariate feature selector.

        Args:
            X: Feature matrix
            y: Target variable
        """
        self.univariate_selector.fit(X, y)
        mask = self.univariate_selector.get_support()
        self.selected_features = list(X.columns[mask])

        # Log selection results
        feature_scores = pd.DataFrame({
            "feature": X.columns,
            "score": self.univariate_selector.scores_,
        })
        feature_scores = feature_scores.sort_values("score", ascending=False)
        logger.info("Top features selected:")
        for _, row in feature_scores.head(self.k).iterrows():
            logger.info(f"{row['feature']}: {row['score']:.4f}")

    def _fit_correlation(
        self, X: pd.DataFrame, y: pd.Series, threshold: float = 0.7
    ) -> None:
        """Fit correlation-based feature selector.

        Args:
            X: Feature matrix
            y: Target variable
            threshold: Correlation threshold for feature removal
        """
        # Add target to get correlations
        data = X.copy()
        data["target"] = y

        # Calculate correlations
        corr_matrix = data.corr().abs()

        # Find features with high correlation with target
        target_corr = corr_matrix["target"].sort_values(ascending=False)

        # Select top k features with highest correlation with target
        self.selected_features = list(target_corr[1 : self.k + 1].index)

        # Log selection results
        logger.info("Top correlated features:")
        for feature, corr in target_corr[1 : self.k + 1].items():
            logger.info(f"{feature}: {corr:.4f}")

    def _log_selection_results(self) -> None:
        """Log feature selection results."""
        if self.feature_scores_ is None:
            return

        feature_scores = pd.DataFrame({
            "feature": self.feature_scores_.keys(),
            "score": self.feature_scores_.values(),
        })
        feature_scores = feature_scores.sort_values("score", ascending=False)

        logger.info("Top selected features:")
        for _, row in feature_scores.head(self.k).iterrows():
            logger.info(f"{row['feature']}: {row['score']:.4f}")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features.

        Args:
            X: Feature matrix

        Returns:
            Transformed feature matrix
        """
        if self.selected_features is None:
            raise ValueError("Selector has not been fitted yet")
        return X[self.selected_features]

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, method: str = "univariate"
    ) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method

        Returns:
            Transformed feature matrix
        """
        return self.fit(X, y, method).transform(X)
