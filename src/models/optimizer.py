"""Model optimization module for the project."""

from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelOptimizer:
    """Class for optimizing model hyperparameters."""

    def __init__(
        self,
        model: BaseEstimator,
        config_path: str = "configs/model_config.yaml",
        random_state: int = 42,
    ):
        """Initialize ModelOptimizer.

        Args:
            model: Base model to optimize
            config_path: Path to model configuration file
            random_state: Random state for reproducibility
        """
        self.model = model
        self.config = load_config(config_path)
        self.random_state = random_state
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.best_model_: Optional[BaseEstimator] = None

    def _get_param_grid(self, model_name: str) -> Dict:
        """Get parameter grid for the specified model.

        Args:
            model_name: Name of the model

        Returns:
            Parameter grid for grid search
        """
        if model_name not in self.config:
            raise ValueError(f"Model {model_name} not found in config")
        return self.config[model_name]

    def optimize(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = "f1",
        n_jobs: int = -1,
        method: str = "grid",
        n_iter: int = 100,
    ) -> BaseEstimator:
        """Optimize model hyperparameters using grid or random search.

        Args:
            X: Feature matrix
            y: Target variable
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            method: Search method ('grid' or 'random')
            n_iter: Number of iterations for random search

        Returns:
            Optimized model
        """
        try:
            logger.info(f"Starting {method} search optimization")

            if param_grid is None:
                model_name = self.model.__class__.__name__.lower()
                param_grid = self._get_param_grid(model_name)

            if method == "grid":
                search = GridSearchCV(
                    self.model,
                    param_grid,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    verbose=1,
                )
            elif method == "random":
                search = RandomizedSearchCV(
                    self.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    random_state=self.random_state,
                    verbose=1,
                )
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            search.fit(X, y)

            self.best_params_ = search.best_params_
            self.best_score_ = search.best_score_
            self.best_model_ = search.best_estimator_

            logger.info("Optimization completed")
            logger.info(f"Best parameters: {self.best_params_}")
            logger.info(f"Best {scoring} score: {self.best_score_:.4f}")

            return self.best_model_

        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            raise

    def get_cv_results(self) -> pd.DataFrame:
        """Get cross-validation results as a DataFrame.

        Returns:
            DataFrame containing CV results
        """
        if not hasattr(self, "best_model_") or self.best_model_ is None:
            raise ValueError("No optimization results available. Run optimize first.")

        cv_results = pd.DataFrame(self.best_model_.cv_results_)
        cv_results = cv_results.sort_values("rank_test_score")

        return cv_results

    def plot_optimization_results(self, param_name: str, ax=None) -> None:
        """Plot optimization results for a specific parameter.

        Args:
            param_name: Name of the parameter to plot
            ax: Matplotlib axis object
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            cv_results = self.get_cv_results()
            param_col = f"param_{param_name}"

            if ax is None:
                _, ax = plt.subplots(figsize=(10, 6))

            sns.scatterplot(data=cv_results, x=param_col, y="mean_test_score", ax=ax)
            ax.set_title(f"Optimization Results for {param_name}")
            ax.set_xlabel(param_name)
            ax.set_ylabel("Mean Test Score")

            plt.tight_layout()

        except Exception as e:
            logger.error(f"Error plotting optimization results: {str(e)}")
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
