"""Script for data preprocessing and feature engineering."""

import argparse
from pathlib import Path

from src.data.loader import load_data
from src.data.preprocessor import DataPreprocessor
from src.features.creator import FeatureCreator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess data and engineer features for customer churn prediction"
        )
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/customer_data.csv",
        help="Path to input data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--feature-config",
        type=str,
        default="configs/feature_config.yaml",
        help="Path to feature configuration",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform detailed data analysis and generate visualizations",
    )
    return parser.parse_args()


def process_and_engineer_features(
    data_path: str,
    output_dir: str,
    feature_config: str,
    analyze: bool = False,
) -> None:
    """Preprocess data and engineer features.

    Args:
        data_path: Path to input data
        output_dir: Directory to save processed data
        feature_config: Path to feature configuration
        analyze: Whether to perform detailed data analysis
    """
    logger.info("Starting data preprocessing and feature engineering pipeline")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        data = load_data(data_path)

        # Initialize preprocessor
        preprocessor = DataPreprocessor()

        # Perform data analysis if requested
        if analyze:
            logger.info("Performing detailed data analysis")
            preprocessor.analyze_data(data, str(output_dir))

        # Preprocess data
        logger.info("Preprocessing data")
        processed_data, scaler = preprocessor.preprocess_data(data)

        # Save preprocessed data
        preprocessed_path = output_dir / "preprocessed_data.csv"
        processed_data.to_csv(preprocessed_path, index=False)
        logger.info(f"Saved preprocessed data to {preprocessed_path}")

        # Initialize feature creator with config
        feature_creator = FeatureCreator(config_path=feature_config)

        # Create features
        logger.info("Engineering features")
        featured_data = feature_creator.create_features(processed_data)

        # Save featured data
        featured_path = output_dir / "featured_data.csv"
        featured_data.to_csv(featured_path, index=False)
        logger.info(f"Saved featured data to {featured_path}")

        # Log feature statistics
        logger.info("\nFeature Engineering Summary:")
        logger.info(f"Original features: {len(data.columns)}")
        logger.info(f"Preprocessed features: {len(processed_data.columns)}")
        logger.info(f"Final features: {len(featured_data.columns)}")

        # Log new features created
        new_features = set(featured_data.columns) - set(processed_data.columns)
        if new_features:
            logger.info("\nNew features created:")
            for feature in sorted(new_features):
                logger.info(f"- {feature}")

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        raise


def main() -> None:
    """Main preprocessing and feature engineering pipeline."""
    try:
        args = parse_args()
        process_and_engineer_features(
            args.data_path,
            args.output_dir,
            args.feature_config,
            args.analyze,
        )
        logger.info(
            "Preprocessing and feature engineering pipeline completed successfully"
        )
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
