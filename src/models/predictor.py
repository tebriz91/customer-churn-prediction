"""Model prediction module for the project."""

from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.utils.logger import get_logger
from src.utils.metrics import calculate_metrics

logger = get_logger(__name__)


class ModelPredictor:
    """Class for making predictions with trained models."""

    def __init__(self, model_path: Optional[str] = None):
        """Initialize ModelPredictor.

        Args:
            model_path: Path to saved model file
        """
        self.model: Optional[BaseEstimator] = None
        self.feature_names: Optional[List[str]] = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        """Load a trained model from file.

        Args:
            model_path: Path to saved model file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _validate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency between training and test data."""
        if not hasattr(self, "feature_names") or not self.feature_names:
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

    def predict(
        self, X: Union[pd.DataFrame, np.ndarray], return_proba: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Make predictions using the loaded model.

        Args:
            X: Feature matrix
            return_proba: Whether to return probability estimates

        Returns:
            Model predictions and optionally probability estimates for positive class
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        try:
            # Validate features if input is DataFrame
            if isinstance(X, pd.DataFrame):
                X = self._validate_features(X)

            predictions = self.model.predict(X)
            if return_proba and hasattr(self.model, "predict_proba"):
                # Return only positive class probabilities (second column)
                probabilities = self.model.predict_proba(X)[:, 1]
                return predictions, probabilities
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray, None] = None,
    ) -> Optional[Dict[str, float]]:
        """Evaluate model performance on test data.

        Args:
            X: Feature matrix
            y: True labels (optional for prediction-only scenarios)

        Returns:
            Dictionary of evaluation metrics if y is provided, None otherwise
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        try:
            predictions = self.predict(X)

            if y is None:
                logger.warning("No target labels provided for evaluation")
                return None

            metrics = calculate_metrics(y, predictions)
            logger.info("Model evaluation metrics:")
            for metric, value in metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def explain_predictions(
        self, X: pd.DataFrame, feature_names: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Explain model predictions using feature importance.

        Args:
            X: Feature matrix
            feature_names: List of feature names

        Returns:
            Dictionary containing feature importance scores
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        try:
            if hasattr(self.model, "feature_importances_"):
                importance_scores = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importance_scores = np.abs(self.model.coef_[0])
            else:
                logger.warning("Model does not support feature importance")
                return {}

            if feature_names is None:
                if hasattr(X, "columns"):
                    feature_names = list(X.columns)
                else:
                    feature_names = [
                        f"feature_{i}" for i in range(len(importance_scores))
                    ]

            importance_dict = dict(zip(feature_names, importance_scores))
            sorted_importance = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )

            logger.info("Top feature importance:")
            for feature, importance in list(sorted_importance.items())[:5]:
                logger.info(f"{feature}: {importance:.4f}")

            return sorted_importance
        except Exception as e:
            logger.error(f"Error explaining predictions: {str(e)}")
            raise


# # Usage example
# from src.models.predictor import ModelPredictor
# from src.models.optimizer import ModelOptimizer
# from sklearn.ensemble import RandomForestClassifier

# # Optimize model
# model = RandomForestClassifier()
# optimizer = ModelOptimizer(model)
# best_model = optimizer.optimize(X_train, y_train)

# # Make predictions
# predictor = ModelPredictor()
# predictor.model = best_model
# predictions = predictor.predict(X_test)
# metrics = predictor.evaluate(X_test, y_test)
