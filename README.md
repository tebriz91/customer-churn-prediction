# Customer Churn Prediction

## Project Structure

```md
beautifulcode/
├── src/                    # Source code
│   ├── data/              # Data processing and validation
│   ├── features/          # Feature engineering and selection
│   ├── models/            # ML models and inference
│   ├── visualization/     # Plotting utilities
│   └── utils/             # Helper functions
├── data/                  # Data files
│   ├── raw/              # Raw input data
│   └── processed/        # Processed and featured data
├── models/                # Saved model artifacts
├── notebooks/             # Jupyter notebooks
├── reports/               # Analysis reports
├── tests/                # Unit and integration tests
├── configs/              # Configuration files
└── scripts/              # Utility scripts
```

## Quick Start

1. Clone the repository
2. Set up the environment (with uv or pip):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

```bash
# Requires uv installed
uv sync
```

3. Data Setup Options:

   a. For Training:
   - Place your training data in `data/raw/`
   - Or use already processed data in `data/processed/`

   b. For Testing/Development:
   - Generate sample test data: `python scripts/create_sample_data.py`
   - This creates `data/new_customers.csv` for testing predictions

4. Training and Prediction Workflow:

```bash
# Process raw training data (if using raw data)
python scripts/preprocess_and_engineer.py --data-path data/raw/customer_data.csv

# Train model on processed data
python scripts/train.py --data-path data/processed/featured_data.csv

# Make predictions on new data
python scripts/predict.py --data-path data/new_customers.csv
```

## Data Requirements

### Required Features

**Numeric Features:**

- Age (18-80)
- Balance (0-100,000)
- EstimatedSalary (30,000-120,000)
- NumOfProducts (1-4)
- Tenure (0-20)
- CreditScore (300-850)
- MonthlyCharges (30-500)
- TotalCharges (1,000-10,000)
- NumTransactions (0-100)

**Categorical Features:**

- Gender (M/F)
- Geography (Mexico/USA)
- HasCrCard (0/1)
- IsActiveMember (0/1)

### Derived Features

The pipeline automatically generates:

- balance_salary_ratio
- products_per_tenure
- active_with_credit_card
- age_group

## Detailed Usage

### 1. Data Preprocessing

```bash
# Required flags: --data-path, --output-dir
python scripts/preprocess_and_engineer.py --data-path data/raw/customer_data.csv --output-dir data/processed

# Optional flag: --analyze to generate analysis plots
python scripts/preprocess_and_engineer.py --data-path data/raw/customer_data.csv --output-dir data/processed --analyze
```

### 2. Model Training

```bash
# Required flags: --data-path, --model-config, --output-dir
python scripts/train.py --data-path data/processed/featured_data.csv --model-config configs/model_config.yaml --output-dir models/
```

Available models:

- Random Forest (`random_forest_model.joblib`)
- Logistic Regression (`logistic_regression_model.joblib`)
- Gradient Boosting (`gradient_boosting_model.joblib`)

### 3. Making Predictions

```bash
# Required flags: --data-path, --model-path, --output-dir
# Optional flag: --evaluate to generate evaluation metrics
python scripts/predict.py --data-path data/new_customers.csv --model-path models/random_forest_model.joblib --output-dir predictions/ --evaluate
```

## Configuration Files

- `configs/model_config.yaml`: Model hyperparameters
- `configs/feature_config.yaml`: Feature engineering settings
- `configs/logging_config.yaml`: Logging settings

## Testing and Quality Assurance

Run the test suite:

```bash
pytest
```

```bash
# With coverage report
pytest --cov=src
```
