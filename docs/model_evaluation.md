# Model Evaluation Documentation

## Overview

This document details the model evaluation pipeline implemented in `model_evaluation.py`. The pipeline provides comprehensive evaluation tools for machine learning models, focusing on performance metrics, visualizations, and error analysis for churn prediction models.

## Dependencies

The evaluation pipeline requires the following Python libraries:

- `pandas` (>= 1.3.0): Data manipulation and analysis
- `numpy` (>= 1.19.0): Numerical operations
- `scikit-learn` (>= 0.24.0): Model evaluation metrics
- `matplotlib` (>= 3.3.0): Visualization
- `seaborn` (>= 0.11.0): Enhanced visualization
- `joblib`: Model loading

## Pipeline Components

### 1. ModelEvaluator Class

The core class that handles all model evaluation operations:

```python
class ModelEvaluator:
    def __init__(self, model_path: str = "models/churn_model.pkl"):
        self.model = joblib.load(model_path)
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = self.model.feature_names_in_
```

### 2. Feature Validation (`_validate_features`)

- **Purpose**: Ensures consistency between training and test features
- **Functionality**:
  - Checks for missing columns
  - Adds missing columns with zeros
  - Removes extra columns
  - Maintains feature order
- **Why**: Prevents errors from feature mismatch

### 3. Model Evaluation (`evaluate_model`)

- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC AUC
- **Output**: Dictionary of metrics and classification report
- **Usage**:

  ```python
  metrics, report = evaluator.evaluate_model(X_test, y_test)
  ```

### 4. Visualization Components

#### Confusion Matrix Plot

- **Function**: `plot_confusion_matrix`
- **Features**:
  - Annotated heatmap
  - Performance metrics overlay
  - Sample size information
  - Custom color scheme
- **Output**: `data/confusion_matrix_eval.png`

#### ROC Curve Plot

- **Function**: `plot_roc_curve`
- **Features**:
  - AUC score display
  - Random baseline comparison
  - Sample size annotation
  - Grid overlay
- **Output**: `data/roc_curve_eval.png`

### 5. Error Analysis (`analyze_errors`)

- **Purpose**: Deep dive into model mistakes
- **Output Features**:
  - Actual vs predicted values
  - Prediction probabilities
  - Confidence scores
  - Error indicators
- **Why**: Helps identify model weaknesses and improvement areas

## Usage Instructions

1. **Basic Usage**:

```python
from beautifulcode.model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator("path/to/model.pkl")

# Evaluate model
metrics, report = evaluator.evaluate_model(X_test, y_test)

# Generate visualizations
evaluator.plot_confusion_matrix(X_test, y_test)
evaluator.plot_roc_curve(X_test, y_test)

# Analyze errors
error_analysis = evaluator.analyze_errors(X_test, y_test)
```

2. **Running the Pipeline**:

```bash
python -m beautifulcode.model_evaluation
```

## Best Practices and Considerations

1. **Data Quality**:
   - Validate feature consistency
   - Check for missing values
   - Ensure proper data types

2. **Visualization**:
   - Consistent figure sizes
   - Clear annotations
   - Proper DPI settings
   - Informative titles and labels

3. **Error Handling**:
   - Comprehensive logging
   - Graceful error recovery
   - Clear error messages

4. **Performance**:
   - Efficient memory usage
   - Proper figure cleanup
   - Resource management

## Testing

### Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=beautifulcode"
```

### Test Components

1. Model loading tests
2. Feature validation tests
3. Metric calculation tests
4. Visualization tests
5. Error analysis tests

## Future Improvements

1. Add support for:
   - Multiple model comparison
   - Custom metrics
   - Interactive visualizations

2. Enhance error analysis:
   - Feature importance in errors
   - Cluster analysis of errors
   - Error patterns visualization

3. Improve scalability:
   - Batch processing
   - Memory-efficient operations
   - Parallel processing

## Troubleshooting

Common issues and solutions:

1. **Model Loading Issues**
   - Error: "Model file not found"
   - Solution: Verify model path and permissions

2. **Feature Mismatch**
   - Problem: Training/test feature inconsistency
   - Solution: Use feature validation method

3. **Memory Issues**
   - Problem: Large dataset visualization
   - Solution: Implement batch processing

4. **Visualization Errors**
   - Issue: Backend compatibility
   - Solution: Configure non-interactive backend

## Output Directory Structure

```md
data/
├── confusion_matrix_eval.png
├── roc_curve_eval.png
└── error_analysis.csv
```

## Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
```

## Error Handling

The evaluation pipeline implements comprehensive error handling:

1. Model loading errors
2. Feature validation errors
3. Visualization errors
4. File I/O errors

Each error is:

- Logged appropriately
- Contains helpful context
- Suggests potential solutions
