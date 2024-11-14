import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd


class ModelInference:
    """Handles model loading and inference for churn prediction."""

    def __init__(self, model_path: Union[str, Path]):
        """Initialize the inference class with a model path.

        Args:
            model_path: Path to the saved model file
        """
        self.model = self._load_model(model_path)
        self.feature_names = (
            self.model.feature_names_in_
            if hasattr(self.model, "feature_names_in_")
            else None
        )

    def _load_model(self, model_path: Union[str, Path]) -> object:
        """Load the model from disk.

        Args:
            model_path: Path to the saved model file

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        try:
            return joblib.load(model_path)
        except FileNotFoundError:
            logging.error(f"Model file not found at {model_path}")
            raise

    def _validate_features(self, data: pd.DataFrame) -> None:
        """Validate that input data has required features.

        Args:
            data: Input DataFrame for prediction

        Raises:
            ValueError: If required features are missing
        """
        if self.feature_names is not None:
            missing_features = set(self.feature_names) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate churn predictions for input data.

        Args:
            data: Input DataFrame containing customer features

        Returns:
            Array of churn predictions

        Raises:
            ValueError: If input data is invalid
        """
        self._validate_features(data)
        try:
            return self.model.predict(data)
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise


def main():
    """Main function to run the inference pipeline."""
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize paths
        model_path = Path("models/churn_model.pkl")
        input_path = Path("data/new_customer_data.csv")
        output_path = Path("data/new_customer_predictions.csv")

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load data
        logging.info("Loading input data...")
        new_data = pd.read_csv(input_path)

        # Initialize inference
        logging.info("Loading model...")
        inference = ModelInference(model_path)

        # Generate predictions
        logging.info("Generating predictions...")
        predictions = inference.predict(new_data)

        # Save results
        new_data["ChurnPrediction"] = predictions
        new_data.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
