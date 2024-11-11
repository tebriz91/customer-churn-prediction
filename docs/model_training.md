# Model Training Documentation

## Overview

This document details the model training pipeline implemented in `model_training.py`. The pipeline is designed to train, optimize, and evaluate machine learning models for customer churn prediction, with a focus on reproducibility and performance monitoring.

## Dependencies

The training pipeline requires the following Python libraries:

- `pandas` (>= 1.3.0): Data manipulation and analysis
- `numpy` (>= 1.19.0): Numerical operations
- `scikit-learn` (>= 0.24.0): Machine learning models and evaluation
- `matplotlib` (>= 3.3.0): Visualization
- `seaborn` (>= 0.11.0): Enhanced visualization
- `joblib`: Model persistence

## Pipeline Components

### 1. ModelTrainer Class

The core class that handles all model training operations:

```python
class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "random_forest": RandomForestClassifier(random_state=random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "logistic_regression": LogisticRegression(random_state=random_state)
        }
```

### 2. Model Training and Evaluation (`train_and_evaluate_models`)

- **Purpose**: Trains and evaluates multiple models using cross-validation
- **Input**: Training features (X_train) and target variable (y_train)
- **Output**: Dictionary with model performances
- **Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Standard deviation of F1-score

### 3. Hyperparameter Optimization (`optimize_model`)

- **Method**: Grid Search with Cross-validation
- **Parameters**:

  ```python
  param_grids = {
      "random_forest": {
          "n_estimators": [100, 200, 300],
          "max_depth": [10, 20, 30, None],
          "min_samples_split": [2, 5, 10],
          "min_samples_leaf": [1, 2, 4],
          "class_weight": ["balanced", None]
      },
      # Additional model parameters...
  }
  ```

- **Optimization Metric**: F1-score
- **Cross-validation**: Adaptive number of splits based on dataset size

### 4. Visualization Components

#### Feature Importance Plot

- **Function**: `plot_feature_importance`
- **Output**: Bar plot of top 10 most important features
- **File**: `data/feature_importance.png`

#### Learning Curves

- **Function**: `plot_learning_curves`
- **Purpose**: Analyze model performance vs training size
- **Output**: `data/learning_curves.png`

#### Confusion Matrix

- **Function**: `plot_confusion_matrix`
- **Output**: Heatmap visualization
- **File**: `data/confusion_matrix.png`

#### ROC Curve

- **Function**: `plot_roc_curve`
- **Output**: ROC curve with AUC score
- **File**: `data/roc_curve.png`

#### Model Comparison

- **Function**: `plot_model_comparison`
- **Output**: Bar plot comparing different models
- **File**: `data/model_comparison.png`

## Usage Instructions

1. **Basic Usage**:

```python
from beautifulcode.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(random_state=42)

# Train and evaluate models
results = trainer.train_and_evaluate_models(X_train, y_train)

# Optimize best model
best_model, best_params = trainer.optimize_model(
    X_train, y_train, model_name="random_forest"
)
```

2. **Running the Pipeline**:

```bash
python -m beautifulcode.model_training
```

## Model Selection and Optimization

### Available Models

1. **Random Forest**
   - Best for: Complex non-linear relationships
   - Advantages: Feature importance, handles outliers well
   - Default parameters: n_estimators=100, random_state=42

2. **Gradient Boosting**
   - Best for: High performance requirements
   - Advantages: Often provides best accuracy
   - Default parameters: n_estimators=100, random_state=42

3. **Logistic Regression**
   - Best for: Baseline model, interpretability
   - Advantages: Fast training, good interpretability
   - Default parameters: max_iter=1000, random_state=42

### Optimization Strategy

1. **Cross-validation**
   - Adaptive splits based on dataset size
   - Minimum 2 splits required
   - Stratified when possible

2. **Grid Search**
   - Exhaustive search over parameter grid
   - Parallel processing with n_jobs=-1
   - F1-score optimization metric

## Error Handling and Logging

### Logging Configuration

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
```

### Key Logged Events

- Model training start/completion
- Cross-validation results
- Hyperparameter optimization progress
- Best parameters and scores
- Visualization generation
- Error conditions

## Best Practices and Considerations

1. **Data Quality**
   - Verify minimum samples for cross-validation
   - Check class balance
   - Validate feature consistency

2. **Model Training**
   - Use stratification when possible
   - Handle class imbalance
   - Monitor for overfitting

3. **Performance**
   - Parallel processing for grid search
   - Efficient memory usage
   - Progress monitoring

4. **Visualization**
   - Consistent figure sizes
   - Clear labels and titles
   - Proper file handling

## Testing

### Test Configuration

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=beautifulcode"
```

### Test Components

1. Model initialization tests
2. Training pipeline tests
3. Optimization tests
4. Visualization tests
5. Error handling tests

## Future Improvements

1. Add support for:
   - Additional models (XGBoost, LightGBM)
   - Custom scoring metrics
   - Advanced optimization techniques (Bayesian optimization)

2. Enhance visualization:
   - Interactive plots
   - Additional performance metrics
   - Customizable plotting options

3. Improve scalability:
   - Distributed training support
   - Memory-efficient processing
   - GPU acceleration

## Troubleshooting

Common issues and solutions:

1. **Insufficient Samples**
   - Error: "Not enough samples for cross-validation"
   - Solution: Ensure at least 4 samples in dataset

2. **Memory Issues**
   - Problem: Grid search memory consumption
   - Solution: Reduce parameter grid or use RandomizedSearchCV

3. **Visualization Errors**
   - Issue: Matplotlib backend errors
   - Solution: Use non-interactive backend (Agg)

4. **Performance Issues**
   - Problem: Slow grid search
   - Solution: Reduce parameter grid or increase n_jobs 