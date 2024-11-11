import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")  # Use non-interactive backend to prevent Tkinter errors

import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    cross_validate,
    learning_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class for training and optimizing machine learning models
    for churn prediction."""

    def __init__(self, random_state: int = 42):
        """Initialize ModelTrainer with configuration parameters."""
        self.random_state = random_state
        self.models = {
            "random_forest": RandomForestClassifier(random_state=random_state),
            "gradient_boosting": GradientBoostingClassifier(
                random_state=random_state,
            ),
            "logistic_regression": LogisticRegression(
                random_state=random_state,
            ),
        }
        self.param_grids = {
            "random_forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "class_weight": ["balanced", None],
            },
            "gradient_boosting": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.3],
                "max_depth": [3, 5, 7],
                "min_samples_split": [2, 5],
                "subsample": [0.8, 0.9, 1.0],
            },
            "logistic_regression": {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "class_weight": ["balanced", None],
                "solver": ["lbfgs", "liblinear"],
                "max_iter": [1000],
            },
        }
        self.best_model = None
        self.feature_importance = None

    def train_and_evaluate_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Train and evaluate multiple models using cross-validation.

        Args:
            X_train: Training features
            y_train: Training target variable

        Returns:
            Dictionary with model performances
        """
        logger.info("Starting model training and evaluation...")

        # Determine number of splits based on dataset size
        n_splits = min(5, len(X_train) // 2)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples for cross-validation. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score, zero_division=0),
            "recall": make_scorer(recall_score, zero_division=0),
            "f1": make_scorer(f1_score, zero_division=0),
        }

        results = {}
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            cv_results = cross_validate(
                model, X_train, y_train, cv=n_splits, scoring=scoring
            )
            results[model_name] = {
                metric: cv_results[f"test_{metric}"].mean() for metric in scoring.keys()
            }
            results[model_name]["std_f1"] = cv_results["test_f1"].std()

            # Break long log message into multiple lines
            logger.info(
                f"{model_name} - "
                f"Accuracy: {results[model_name]['accuracy']:.3f}, "
                f"Precision: {results[model_name]['precision']:.3f}, "
                f"Recall: {results[model_name]['recall']:.3f}, "
                f"F1: {results[model_name]['f1']:.3f} "
                f"(±{results[model_name]['std_f1']:.3f})"
            )

        return results

    def optimize_model(
        self, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Perform hyperparameter optimization for the selected model."""
        logger.info(f"Starting hyperparameter optimization for {model_name}...")

        model = self.models[model_name]
        param_grid = self.param_grids[model_name]

        # Determine number of splits based on dataset size
        n_splits = min(5, len(X_train) // 2)
        if n_splits < 2:
            raise ValueError(
                f"Not enough samples for cross-validation. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        # Add error handling for grid search
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    message="invalid value encountered in cast",
                )
                grid_search = GridSearchCV(
                    model, param_grid, cv=n_splits, scoring="f1", n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error during grid search: {str(e)}")
            raise

        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")

        self.best_model = grid_search.best_estimator_

        if hasattr(self.best_model, "feature_importances_"):
            self.feature_importance = pd.DataFrame({
                "feature": X_train.columns,
                "importance": self.best_model.feature_importances_,
            }).sort_values("importance", ascending=False)

        return grid_search.best_estimator_, grid_search.best_params_

    def plot_feature_importance(self, output_dir: str = "data"):
        """Plot feature importance if available."""
        if self.feature_importance is None:
            logger.warning("Feature importance not available for this model")
            return

        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.feature_importance.head(10), x="importance", y="feature")
        plt.title("Top 10 Most Important Features")
        plt.tight_layout()

        output_path = Path(output_dir) / "feature_importance.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Feature importance plot saved to {output_path}")

    def save_model(self, model_path: str = "models/churn_model.pkl"):
        """Save the trained model to disk."""
        if self.best_model is None:
            raise ValueError("No model has been trained yet")

        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_model, model_path)
        logger.info(f"Model saved to {model_path}")

    def plot_learning_curves(
        self, X_train: pd.DataFrame, y_train: pd.Series, output_dir: str = "data"
    ) -> None:
        """Plot learning curves to analyze model performance vs training size."""
        if self.best_model is None:
            logger.warning("No model has been trained yet")
            return

        # Check if we have both classes in the dataset
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            logger.warning(
                "Cannot generate learning curves: need samples from both classes"
            )
            return

        # Determine appropriate number of splits based on dataset size
        n_splits = min(3, len(X_train) // 2)
        if n_splits < 2:
            logger.warning(
                f"Not enough samples for learning curves. "
                f"Got {len(X_train)} samples, need at least 4."
            )
            return

        # Ensure minimum samples per class
        class_counts = pd.Series(y_train).value_counts()
        min_samples = class_counts.min()
        if min_samples < 2:
            logger.warning(
                f"Not enough samples per class. Minimum required: 2, got {min_samples}"
            )
            return

        try:
            # Calculate appropriate train sizes
            min_size = max(2, int(len(X_train) * 0.5))
            max_size = len(X_train)
            train_size_range = [0.5, 1.0]

            train_sizes, train_scores, val_scores = learning_curve(
                self.best_model,
                X_train,
                y_train,
                train_sizes=train_size_range,
                cv=n_splits,
                scoring="accuracy",  # Changed from f1 to accuracy
                n_jobs=-1,
                error_score="raise",
            )

            plt.figure(figsize=(10, 6))
            train_mean = np.mean(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)

            plt.plot(train_sizes, train_mean, label="Training score")
            plt.plot(train_sizes, val_mean, label="Cross-validation score")
            plt.xlabel("Training Examples")
            plt.ylabel("Accuracy Score")  # Changed from F1 to Accuracy
            plt.title("Learning Curves")
            plt.legend(loc="best")
            plt.grid(True)

            output_path = Path(output_dir) / "learning_curves.png"
            plt.savefig(output_path)
            plt.close()

            logger.info(f"Learning curves plot saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not generate learning curves: {str(e)}")

    def plot_confusion_matrix(
        self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "data"
    ) -> None:
        """Plot confusion matrix for model predictions."""
        if self.best_model is None:
            logger.warning("No model has been trained yet")
            return

        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        output_path = Path(output_dir) / "confusion_matrix.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Confusion matrix plot saved to {output_path}")

    def plot_roc_curve(
        self, X_test: pd.DataFrame, y_test: pd.Series, output_dir: str = "data"
    ) -> None:
        """Plot ROC curve for model predictions."""
        if self.best_model is None:
            logger.warning("No model has been trained yet")
            return

        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        output_path = Path(output_dir) / "roc_curve.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"ROC curve plot saved to {output_path}")

    def plot_model_comparison(
        self, results: Dict[str, Dict[str, float]], output_dir: str = "data"
    ) -> None:
        """Plot comparison of different models' performance metrics."""
        metrics = ["accuracy", "precision", "recall", "f1"]
        model_names = list(results.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.2
        multiplier = 0

        for metric in metrics:
            metric_values = [results[model][metric] for model in model_names]
            offset = width * multiplier
            ax.bar(x + offset, metric_values, width, label=metric.capitalize())
            multiplier += 1

        ax.set_ylabel("Scores")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = Path(output_dir) / "model_comparison.png"
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Model comparison plot saved to {output_path}")


def main():
    """Main function to run the model training pipeline."""
    try:
        # Load the preprocessed data
        logger.info("Loading preprocessed data...")
        X_train = pd.read_csv("data/X_train_featured.csv")
        y_train = pd.read_csv("data/y_train.csv").values.ravel()

        # Load test data for visualization
        X_test = pd.read_csv("data/X_test_featured.csv")
        y_test = pd.read_csv("data/y_test.csv").values.ravel()

        # Verify data was loaded correctly
        if len(X_train) < 4:
            raise ValueError(
                "Not enough samples for training. "
                f"Got {len(X_train)} samples, need at least 4."
            )

        logger.info(f"Loaded {len(X_train)} training samples")

        # Ensure consistent features between train and test sets
        all_features = sorted(list(set(X_train.columns) | set(X_test.columns)))

        # Add missing columns with zeros to both sets
        for col in all_features:
            if col not in X_train.columns:
                X_train[col] = 0
            if col not in X_test.columns:
                X_test[col] = 0

        # Ensure same column order
        X_train = X_train[all_features]
        X_test = X_test[all_features]

        logger.info("Aligned features between train and test sets")

        # Initialize trainer
        trainer = ModelTrainer(random_state=42)

        # Evaluate all models
        results = trainer.train_and_evaluate_models(X_train, y_train)

        # Plot model comparison
        trainer.plot_model_comparison(results)

        # Select the best performing model type based on F1 score
        best_model_name = max(results, key=lambda k: results[k]["f1"])
        logger.info(f"Best performing model: {best_model_name}")

        # Optimize the best model
        best_model, best_params = trainer.optimize_model(
            X_train, y_train, best_model_name
        )

        # Generate visualization plots
        trainer.plot_feature_importance()

        # Only generate additional plots if we have enough data
        if len(X_train) >= 4:
            trainer.plot_learning_curves(X_train, y_train)
            trainer.plot_confusion_matrix(X_test, y_test)
            trainer.plot_roc_curve(X_test, y_test)
        else:
            logger.warning("Skipping some visualizations due to insufficient data")

        # Save the model
        trainer.save_model()

        logger.info("Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
