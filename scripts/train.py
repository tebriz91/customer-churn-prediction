"""Training script for the customer churn prediction model."""

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_data
from src.data.preprocessor import DataPreprocessor
from src.features.creator import FeatureCreator
from src.models.trainer import ModelTrainer
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.metrics import generate_classification_report
from src.visualization.evaluation import ModelEvaluationPlotter

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train customer churn prediction model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/customer_data.csv",
        help="Path to input data",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model artifacts",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for reproducibility",
    )
    return parser.parse_args()


def prepare_data(
    data_path: str, random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for training.

    Args:
        data_path: Path to input data
        random_state: Random state for reproducibility

    Returns:
        Train and test splits of features and target
    """
    logger.info("Loading and preparing data")

    # Load and preprocess data
    data = load_data(data_path)
    preprocessor = DataPreprocessor()
    processed_data, scaler = preprocessor.preprocess_data(data)

    # Create features
    feature_creator = FeatureCreator()
    featured_data = feature_creator.create_features(processed_data)

    # Split features and target
    X = featured_data.drop("Churn", axis=1)
    y = featured_data["Churn"]

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config_path: str,
    output_dir: str,
    random_state: int,
) -> None:
    """Train and evaluate models.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        config_path: Path to model configuration
        output_dir: Directory to save model artifacts
        random_state: Random state for reproducibility
    """
    logger.info("Training and evaluating models")

    # Load model configuration
    config = load_config(config_path)

    # Initialize trainer and train models
    trainer = ModelTrainer(random_state=random_state)
    models = trainer.train_and_evaluate_models(X_train, y_train)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize visualization
    eval_plotter = ModelEvaluationPlotter()

    # Evaluate each model
    for name, model_info in models.items():
        model = model_info["model"]

        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Generate and save evaluation report
        report = generate_classification_report(y_test, y_pred, y_prob)

        # Save evaluation plots
        eval_plotter.plot_confusion_matrix(
            y_test,
            y_pred,
            title=f"Confusion Matrix - {name}",
            save_path=output_dir / f"{name}_confusion_matrix.png",
        )

        eval_plotter.plot_roc_curve(
            y_test,
            y_prob,
            title=f"ROC Curve - {name}",
            save_path=output_dir / f"{name}_roc_curve.png",
        )

        eval_plotter.plot_precision_recall_curve(
            y_test,
            y_prob,
            title=f"Precision-Recall Curve - {name}",
            save_path=output_dir / f"{name}_pr_curve.png",
        )

        # Save model and metrics
        model_path = output_dir / f"{name}_model.joblib"
        trainer.save_model(model, model_path)

        logger.info(f"\nEvaluation results for {name}:")
        for metric, value in report["classification_metrics"].items():
            logger.info(f"{metric}: {value:.4f}")


def main() -> None:
    """Main training pipeline."""
    try:
        # Parse arguments
        args = parse_args()

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(
            args.data_path, args.random_state
        )

        # Train and evaluate models
        train_and_evaluate(
            X_train,
            X_test,
            y_train,
            y_test,
            args.model_config,
            args.output_dir,
            args.random_state,
        )

        logger.info("Training pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()