# Data Preprocessing Documentation

## Overview

This document details the data preprocessing pipeline implemented in `data_preprocessing.py`. The pipeline is designed to prepare customer data for a churn prediction model, ensuring data quality and consistency while maintaining reproducibility.

## Dependencies

The preprocessing pipeline requires the following Python libraries:

- `pandas` (>= 1.3.0): Data manipulation and analysis
- `numpy` (>= 1.19.0): Numerical operations
- `scikit-learn` (>= 0.24.0): Data preprocessing and splitting
- `matplotlib` (>= 3.3.0): Data visualization
- `seaborn` (>= 0.11.0): Enhanced data visualization

## Pipeline Components

### 1. Data Loading (`load_data`)

```python
def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)
```

- **Purpose**: Loads customer data from a CSV file
- **Input**: Path to CSV file
- **Output**: Pandas DataFrame
- **Why**: Simple function to ensure consistent data loading across the pipeline

### 2. Data Analysis (`analyze_data`)

- **Purpose**: Performs exploratory data analysis (EDA) and generates visualizations

- **Key Components**:
  - Basic statistics (mean, std, quartiles)
  - Missing value analysis
  - Correlation matrix visualization
  - Distribution plots for numeric features

- **Outputs**:
  - `data/correlation_matrix.png`: Heatmap showing feature correlations
  - `data/numeric_distributions.png`: Distribution plots for numeric features

- **Why**: Understanding data distributions and relationships is crucial for:
  - Identifying potential data quality issues
  - Informing feature engineering decisions
  - Detecting potential biases in the dataset

### 3. Missing Value Handling (`handle_missing_values`)

- **Strategy**:
  - Numeric columns: Median imputation using `df.assign()` to avoid chained assignment warnings
  - Categorical columns: Mode imputation
- **Why This Approach**:
  - Median is robust to outliers compared to mean
  - Mode preserves the most common category for categorical data
  - Both methods maintain data distribution characteristics
  - Using `df.assign()` prevents pandas FutureWarning about chained assignments
- **Logging**: Tracks missing value counts before and after imputation

### 4. Outlier Handling (`handle_outliers`)

- **Method**: IQR (Interquartile Range) method

- **Parameters**:
  - IQR = Q3 - Q1
  - Lower bound = Q1 - 1.5 * IQR
  - Upper bound = Q3 + 1.5 * IQR

- **Why IQR Method**:
  - Robust statistical method
  - Less sensitive to extreme values
  - Preserves data distribution while removing extreme outliers

- **Treatment**: Values outside bounds are capped rather than removed

- **Why Capping**:
  - Preserves data points
  - Maintains sample size
  - Reduces impact of extreme values

### 5. Main Preprocessing Pipeline (`preprocess_data`)

Sequential steps:

1. Data analysis
2. Missing value handling
3. Outlier treatment
4. Categorical variable encoding
5. Feature scaling

**Feature Scaling**:

- Uses StandardScaler
- Why: Ensures all features contribute equally to model training
- Returns both transformed data and scaler for consistent transformation of new data

### 6. Data Splitting (`split_data`)

- **Parameters**:
  - `test_size=0.2`: 80-20 train-test split
  - `random_state=42`: Ensures reproducibility
  - Uses stratification on target variable when possible

- **Stratification Handling**:
  - Automatically checks if there are enough samples per class (minimum 2)
  - Falls back to regular random split if stratification is not possible
  - Provides clear warning message when stratification cannot be used
  - Maintains consistent line length in warning messages

- **Output Information**:
  - Reports train and test set sizes
  - Shows target distribution in both train and test sets
  - Formats output messages to maintain readability and line length limits

## Usage Instructions

1. **Setup**:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

1. **Running the Pipeline**:

```bash
python data_preprocessing.py
```

1. **Expected Directory Structure**:

```md
project/
├── data/
│ ├── customer_data.csv
│ ├── correlation_matrix.png
│ ├── numeric_distributions.png
│ ├── X_train.csv
│ ├── X_test.csv
│ ├── y_train.csv
│ └── y_test.csv
├── data_preprocessing.py
└── docs/
└── data_preprocessing.md
```

1. **Output Files**:

- Processed datasets in `data/` directory
- Visualization files for analysis
- Console logs detailing preprocessing steps

## Best Practices and Considerations

1. **Data Quality**:
   - Always check data types before processing
   - Validate assumptions about value ranges
   - Monitor changes in data distribution
   - Handle small datasets appropriately with fallback options

2. **Reproducibility**:
   - Fixed random seed (42)
   - Documented parameter choices
   - Preserved preprocessing steps for new data

3. **Performance**:
   - Efficient pandas operations
   - Minimal memory footprint
   - Scalable to larger datasets

4. **Monitoring**:
   - Detailed logging at each step
   - Data quality metrics
   - Visual validation options

## Customization

The pipeline can be customized by modifying:

- Outlier thresholds (currently 1.5 * IQR)
- Train-test split ratio (currently 0.2)
- Feature scaling method
- Categorical encoding strategy

## Troubleshooting

Common issues and solutions:

1. **Missing data directory**: Create `data/` directory before running
2. **Memory issues**: Process data in chunks for large datasets
3. **Scaling errors**: Ensure numeric columns are properly typed
4. **Visualization errors**: Check matplotlib backend configuration
5. **Stratification errors**: Will automatically handle small datasets by falling back to regular splitting

## Future Improvements

Potential enhancements:

1. Add support for automated feature selection
2. Implement more sophisticated imputation methods
3. Add cross-validation support
4. Enhance visualization capabilities
5. Add data validation checks

## Testing

### Test Configuration

The project uses pytest with coverage reporting. Configuration is defined in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=beautifulcode"
pythonpath = ["src"]
```

### Running Tests

To run the test suite:

```bash
pytest
```

This will:

- Run all tests with verbose output (-v)
- Generate coverage report for beautifulcode package
- Use test discovery in the tests/ directory

### Test Components

The test suite in `tests/test_data_preprocessing.py` includes:

1. **Data Loading Tests**
   - Validates CSV file loading
   - Handles missing file errors
   - Tests error conditions

2. **Data Analysis Tests**
   - Verifies visualization generation
   - Checks statistical calculations
   - Tests output directory creation

3. **Missing Value Tests**
   - Validates median imputation for numeric columns
   - Tests mode imputation for categorical columns
   - Verifies complete handling of missing values

4. **Outlier Tests**
   - Tests IQR-based outlier detection
   - Verifies outlier capping
   - Validates statistics calculation

5. **Preprocessing Pipeline Tests**
   - Tests complete data preprocessing workflow
   - Validates scaling results
   - Checks categorical encoding

6. **Data Splitting Tests**
   - Tests train/test splitting
   - Validates stratification
   - Checks different test size ratios

### Test Coverage

Coverage reporting is handled automatically through pytest-cov. The coverage report shows:
- Line coverage
- Branch coverage
- Missing lines
- Overall coverage percentage

### Test Fixtures

The test suite uses a sample dataset fixture that provides:
- Realistic customer data structure
- Known missing values
- Controlled outliers
- Both numeric and categorical features

### Best Practices

1. **Test Independence**
   - Each test can run independently
   - Tests don't rely on external data
   - Clean state between tests

2. **Comprehensive Testing**
   - Tests both success and error paths
   - Covers edge cases
   - Validates all main functionality

3. **Clear Test Names**
   - Descriptive test function names
   - Clear test purpose documentation
   - Well-organized test structure

4. **Assertions**
   - Specific assertion messages
   - Appropriate tolerance for floating-point comparisons
   - Validates both data types and values

### Maintaining Tests

When modifying the preprocessing pipeline:
1. Run the full test suite
2. Update tests for new functionality
3. Maintain test coverage
4. Update test documentation

### Common Test Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_data_preprocessing.py

# Run specific test
pytest tests/test_data_preprocessing.py::test_load_data

# Show coverage report
pytest --cov-report term-missing
```
