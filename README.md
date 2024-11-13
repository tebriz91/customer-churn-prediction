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

1. Data Preprocessing:

```bash
python scripts/train.py
```

2. Model Training:

```bash
python scripts/train.py --data-path data/customer_data.csv --model-config configs/model_config.yaml --output-dir models/
```

3. Making Predictions:

```bash
python scripts/predict.py --data-path data/new_customers.csv --model-path models/random_forest_model.joblib --output-dir predictions/
```

4. Making Predictions with Evaluation:

```bash
python scripts/predict.py --data-path data/test_data.csv --model-path models/best_model.joblib --output-dir predictions/ --evaluate
```

## Configuration

- Model parameters: `configs/model_config.yaml`
- Feature settings: `configs/feature_config.yaml`
- Logging settings: `configs/logging_config.yaml`

## Testing

Run tests with:

```bash
pytest
```
