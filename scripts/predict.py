"""Prediction script for the customer churn prediction model."""

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_data
from src.data.preprocessor import DataPreprocessor
from src.features.creator import FeatureCreator
from src.models.predictor import ModelPredictor
from src.utils.logger import get_logger
from src.utils.metrics import generate_classification_report
from src.visualization.evaluation import ModelEvaluationPlotter

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Make customer churn predictions")
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input data",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory to save prediction results",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate predictions if true labels are available",
    )
    return parser.parse_args()


def prepare_prediction_data(data_path: str) -> pd.DataFrame:
    """Prepare data for prediction.

    Args:
        data_path: Path to input data

    Returns:
        Processed features ready for prediction
    """
    logger.info("Loading and preparing data for prediction")

    # Load and preprocess data
    data = load_data(data_path)
    preprocessor = DataPreprocessor()
    processed_data, _ = preprocessor.preprocess_data(data)

    # Create features
    feature_creator = FeatureCreator()
    featured_data = feature_creator.create_features(processed_data)

    # Prepare features
    if "Churn" in featured_data.columns:
        X = featured_data.drop("Churn", axis=1)
    else:
        X = featured_data

    return X


def make_predictions(
    X: pd.DataFrame,
    model_path: str,
    output_dir: str,
    true_labels: pd.Series = None,
) -> None:
    """Make and optionally evaluate predictions.

    Args:
        X: Input features
        model_path: Path to trained model
        output_dir: Directory to save results
        true_labels: True labels for evaluation (optional)
    """
    logger.info("Making predictions")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and make predictions
    predictor = ModelPredictor(model_path)
    predictions = predictor.predict(X)
    probabilities = predictor.predict(X, return_proba=True)[1]

    # Save predictions
    results = pd.DataFrame({"prediction": predictions, "probability": probabilities})
    results.to_csv(output_dir / "predictions.csv", index=False)
    logger.info(f"Saved predictions to {output_dir}/predictions.csv")

    # Evaluate if true labels are available
    if true_labels is not None:
        logger.info("Evaluating predictions")

        # Generate evaluation report
        report = generate_classification_report(true_labels, predictions, probabilities)

        # Create evaluation plots
        eval_plotter = ModelEvaluationPlotter()

        eval_plotter.plot_confusion_matrix(
            true_labels,
            predictions,
            save_path=output_dir / "confusion_matrix.png",
        )

        eval_plotter.plot_roc_curve(
            true_labels,
            probabilities,
            save_path=output_dir / "roc_curve.png",
        )

        eval_plotter.plot_precision_recall_curve(
            true_labels,
            probabilities,
            save_path=output_dir / "pr_curve.png",
        )

        # Log evaluation metrics
        logger.info("\nEvaluation Results:")
        for metric, value in report["classification_metrics"].items():
            logger.info(f"{metric}: {value:.4f}")


def main() -> None:
    """Main prediction pipeline."""
    try:
        # Parse arguments
        args = parse_args()

        # Prepare data
        data = prepare_prediction_data(args.data_path)

        # Get true labels if evaluation is requested
        true_labels = None
        if args.evaluate:
            raw_data = pd.read_csv(args.data_path)
            if "Churn" in raw_data.columns:
                true_labels = raw_data["Churn"]
            else:
                logger.warning("No 'Churn' column found for evaluation")

        # Make predictions
        make_predictions(
            data,
            args.model_path,
            args.output_dir,
            true_labels,
        )

        logger.info("Prediction pipeline completed successfully")

    except Exception as e:
        logger.error(f"Error in prediction pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
