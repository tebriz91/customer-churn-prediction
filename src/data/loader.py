import os

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file.

    Args:
        file_path: Path to the CSV file to load

    Returns:
        Loaded DataFrame

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If there's an error loading the file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        logger.info(f"DataFrame shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise RuntimeError(f"Error loading data: {str(e)}")


def save_split_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: str = "data",
) -> None:
    """Save split datasets to CSV files.

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        output_dir: Directory to save files
    """
    logger.info("Saving processed data...")
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    logger.info(f"Saved processed datasets to {output_dir}/")
