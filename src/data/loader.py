import os

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")
