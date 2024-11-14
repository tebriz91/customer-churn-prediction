# Beautiful Code Hackathon - Customer Churn Prediction

## Project Overview

This project implements a machine learning pipeline for predicting customer churn. It includes data preprocessing, feature engineering, model training, and evaluation components.

## Project Structure

```md
beautifulcode/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Helper functions
├── data/                  # Data files
├── models/                # Saved models
├── notebooks/             # Jupyter notebooks
├── reports/               # Analysis reports
├── tests/                 # Test files
├── configs/               # Configuration files
└── scripts/               # Utility scripts
```

## Setup

1. Clone the repository
2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -e .
```

## Usage

1. Data Preprocessing and Feature Engineering:

```bash
# Basic preprocessing and feature engineering
python scripts/preprocess_and_engineer.py --data-path data/raw/customer_data.csv

# With data analysis and visualizations
python scripts/preprocess_and_engineer.py --data-path data/raw/customer_data.csv --analyze

# With custom output directory and feature configuration
python scripts/preprocess_and_engineer.py \
    --data-path data/raw/customer_data.csv \
    --output-dir data/custom_processed \
    --feature-config configs/custom_feature_config.yaml
```

2. Model Training:

```bash
python scripts/train.py --data-path data/processed/featured_data.csv --model-config configs/model_config.yaml --output-dir models/
```

3. Making Predictions:

```bash
python scripts/predict.py --data-path data/new_customers.csv \
    --model-path models/random_forest_model.joblib \
    --output-dir predictions/
```

4. Making Predictions with Evaluation:

```bash
python scripts/predict.py --data-path data/test_data.csv \
    --model-path models/best_model.joblib \
    --output-dir predictions/ \
    --evaluate
```

## Configuration

The project uses several configuration files:

- `configs/model_config.yaml`: Model hyperparameters and training settings
- `configs/feature_config.yaml`: Feature engineering settings and parameters
- `configs/logging_config.yaml`: Logging configuration

## Data Pipeline

1. **Preprocessing** (`preprocess_and_engineer.py`):
   - Data cleaning and validation
   - Missing value handling
   - Outlier detection
   - Feature scaling
   - Categorical encoding

2. **Feature Engineering**:
   - Automated feature creation
   - Feature selection
   - Feature validation
   - Data quality checks

3. **Model Training** (`train.py`):
   - Model selection
   - Hyperparameter optimization
   - Cross-validation
   - Model evaluation

4. **Prediction** (`predict.py`):
   - Batch predictions
   - Model evaluation
   - Results visualization

## Testing

Run tests with:

```bash
pytest
```

## Documentation

Detailed documentation for each component can be found in the `docs/` directory.
