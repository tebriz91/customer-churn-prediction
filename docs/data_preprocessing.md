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

The preprocessing pipeline includes a comprehensive test suite with the following updates:

### Test Components

#### 1. Test Fixtures

- `sample_data`: Creates a realistic dataset with known characteristics including:
  - Numeric and categorical features
  - Controlled missing values
  - Known outliers
  - Balanced and imbalanced class distributions

#### 2. Basic Functionality Tests

- `test_load_data`: Validates CSV data loading
- `test_analyze_data`: Checks data analysis and visualization generation
- `test_handle_missing_values`: Verifies missing value imputation strategies
- `test_handle_outliers`: Ensures proper outlier detection and handling

#### 3. Pipeline Tests

- `test_preprocess_data`: Validates the complete preprocessing pipeline
- `test_split_data`: Checks data splitting functionality
- `test_split_data_stratification`: Verifies stratified splitting for imbalanced data

#### 4. Quality Assurance Tests

- `test_data_leakage`: Ensures no data leakage between train and test sets
- `test_reproducibility`: Validates result reproducibility with fixed random states

#### 5. Code Quality Tests

- **Line Length Compliance**:
  - All code lines are limited to 79 characters
  - Long strings are properly formatted using line continuation
  - Warning and error messages are wrapped appropriately

- **Standard Deviation Tests**:
  - Numeric scaling tests use appropriate precision (1e-3)
  - Mean tests use appropriate precision (1e-5)
  - Clear error messages for failed assertions

### Running Tests

1. Install test dependencies with uv:

```bash
# Install main dependencies
uv pip install -e .

# Install test dependencies
uv pip install pytest pytest-cov
```

2. Execute the test suite:

```bash
# Run all tests with coverage
pytest -v --cov=beautifulcode

# Run specific test file
pytest tests/test_data_preprocessing.py -v

# Run with detailed coverage report
pytest -v --cov=beautifulcode --cov-report=term-missing
```

The project uses the following test configuration in pyproject.toml:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=beautifulcode"
pythonpath = ["src"]
```

### Test Best Practices

The test suite follows these best practices:

1. **Test Data Management**
   - Uses fixtures for consistent test data
   - Includes edge cases and boundary conditions
   - Simulates real-world data scenarios

2. **Quality Assurance**
   - Checks for data leakage
   - Ensures reproducibility
   - Validates data integrity throughout the pipeline

3. **Test Coverage**
   - Tests both success and error paths
   - Includes parameterized tests for different scenarios
   - Covers all main functionality components

4. **Documentation**
   - Clear test names and descriptions
   - Documented test fixtures and utilities
   - Example usage and expected outcomes

### Test Maintenance

When modifying the preprocessing pipeline:

1. Run the full test suite to ensure no regressions
2. Add tests for new functionality
3. Update existing tests if requirements change
4. Maintain test documentation

### Continuous Integration

The test suite is designed to be run in CI/CD pipelines with:

- Automated test execution
- Coverage reporting
- Linting checks
- Documentation validation

For detailed test implementation, refer to `tests/test_data_preprocessing.py`.
