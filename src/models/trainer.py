import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from lime import lime_tabular
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    cross_validate,
)

from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """Class for training machine learning models for churn prediction."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize ModelTrainer with configuration parameters."""
        self.config = config or Config()
        self.random_state = self.config.model.random_state
        self.models = {
            "random_forest": RandomForestClassifier(random_state=self.random_state),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=self.random_state
            ),
            "logistic_regression": LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=-1,
            ),
        }
        self.param_grids = self.config.model.param_grids
        self.best_model = None

    def train_and_evaluate_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """Train and evaluate multiple models using cross-validation."""
        logger.info("Starting model training and evaluation...")

        n_splits = min(5, len(X_train) // 2)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples for cross-validation. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        scoring = {"f1": make_scorer(f1_score)}

        results = {}
        for model_name, model in self.models.items():
            logger.info(f"\nTraining {model_name}...\n")
            cv_results = cross_validate(
                model, X_train, y_train, cv=n_splits, scoring=scoring
            )
            results[model_name] = {
                "f1": cv_results["test_f1"].mean(),
                "std_f1": cv_results["test_f1"].std(),
            }

            logger.info(
                f"{model_name} - F1: {results[model_name]['f1']:.3f} "
                f"(±{results[model_name]['std_f1']:.3f})"
            )

        return results

    def optimize_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Perform hyperparameter optimization for the selected model."""
        logger.info(f"Starting optimization for {model_name}...")

        model = self.models[model_name]
        param_grid = self.param_grids[model_name]

        n_splits = min(5, len(X_train) // 2)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples for cross-validation. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                logger.debug(f"Running GridSearchCV with parameters: {param_grid}")
                grid_search = GridSearchCV(
                    model, param_grid, cv=n_splits, scoring="f1", n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error during grid search: {str(e)}")
            raise

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.3f}")

        self.best_model = grid_search.best_estimator_
        return grid_search.best_estimator_, grid_search.best_params_

    def save_model(self, model_path: Path) -> None:
        """Save model to disk.

        Args:
            model_path: Path to save the model
        """
        try:
            joblib.dump(self.best_model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def explain_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Explain the model using LIME."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")

        logger.info("Calculating LIME explanations for model explainability...")

        # Create a LIME explainer
        explainer = lime_tabular.LimeTabularExplainer(
            X_train.values,
            feature_names=X_train.columns,
            class_names=["Not Churn", "Churn"],
            discretize_continuous=True,
        )

        # Explain a single instance (e.g., the first instance)
        i = 0
        logger.debug(f"Explaining instance at index {i}")
        exp = explainer.explain_instance(
            X_train.iloc[i].values, self.best_model.predict_proba, num_features=10
        )

        # Print the explanation to the console
        logger.info("LIME explanation for instance index: %d", i)
        for feature, weight in exp.as_list():
            logger.info(f"{feature}: {weight:.4f}")


def main():
    """Main function to run the model training pipeline."""
    try:
        # Load the preprocessed data
        logger.info("Loading preprocessed data...")
        X_train = pd.read_csv("data/X_train_featured.csv")
        y_train = pd.read_csv("data/y_train.csv").values.ravel()

        # Verify data was loaded correctly
        if len(X_train) < 4:
            raise ValueError(
                f"Not enough samples for training. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        logger.info(f"Loaded {len(X_train)} training samples")

        # Initialize trainer
        trainer = ModelTrainer(config=Config())
        logger.debug("Initialized ModelTrainer")

        # Train and evaluate models
        results = trainer.train_and_evaluate_models(X_train, y_train)
        logger.debug(f"Training results: {results}")

        # Select the best performing model type based on F1 score
        best_model_name = max(results, key=lambda k: results[k]["f1"])
        logger.info(f"Best performing model: {best_model_name}")

        # Optimize the best model
        trainer.optimize_model(X_train, y_train, best_model_name)

        # Save the model
        trainer.save_model(Path("models/churn_model.pkl"))

        # Explain the best model
        trainer.explain_model(X_train, y_train)

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
