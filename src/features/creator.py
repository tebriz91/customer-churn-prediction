from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureCreator:
    def __init__(self):
        self.preprocessor = DataPreprocessor()

    def create_features(self, df):
        """Create new features"""
        # ... existing feature creation code ...
