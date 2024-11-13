import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # ... existing handle_missing_values code ...

    def handle_outliers(self, df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        # ... existing handle_outliers code ...

    def preprocess_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        """Preprocess data for modeling"""
        # ... existing preprocess_data code ...
