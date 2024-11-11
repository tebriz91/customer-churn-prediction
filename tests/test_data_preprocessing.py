import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from beautifulcode.data_preprocessing import (
    analyze_data,
    handle_missing_values,
    handle_outliers,
    load_data,
    preprocess_data,
    split_data,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    n_samples = 1000

    data = {
        "Age": np.random.normal(40, 10, n_samples),
        "Balance": np.random.normal(50000, 20000, n_samples),
        "Gender": np.random.choice(["M", "F"], n_samples),
        "Tenure": np.random.randint(0, 10, n_samples, dtype=np.int64),
        "NumOfProducts": np.random.randint(1, 4, n_samples, dtype=np.int64),
        "HasCrCard": np.random.choice([0, 1], n_samples).astype(np.int64),
        "IsActiveMember": np.random.choice([0, 1], n_samples).astype(np.int64),
        "EstimatedSalary": np.random.normal(70000, 30000, n_samples),
        "Churn": np.random.choice([0, 1], n_samples).astype(np.int64),
    }

    # Create DataFrame first
    df = pd.DataFrame(data)

    # Add missing values to specific columns
    for col in ["Age", "Balance", "Gender"]:
        mask = np.random.choice([True, False], n_samples, p=[0.05, 0.95])
        df.loc[mask, col] = np.nan

    return df


def test_load_data(tmp_path):
    """Test data loading functionality"""
    # Create a temporary CSV file
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    # Test loading
    loaded_df = load_data(file_path)
    assert isinstance(loaded_df, pd.DataFrame)
    assert loaded_df.shape == (3, 2)
    assert all(loaded_df.columns == ["A", "B"])


def test_analyze_data(sample_data, tmp_path):
    """Test data analysis functionality"""
    # Configure matplotlib to use non-interactive backend
    import matplotlib

    matplotlib.use("Agg")

    # Run analysis with tmp_path
    analyzed_df = analyze_data(sample_data, output_dir=str(tmp_path))

    # Check if visualization files were created in tmp_path
    assert (tmp_path / "correlation_matrix.png").exists()
    assert (tmp_path / "numeric_distributions.png").exists()

    # Check if DataFrame was returned unchanged
    assert analyzed_df.equals(sample_data)


def test_handle_missing_values(sample_data):
    """Test missing values handling"""
    # Get initial missing values count
    initial_missing = sample_data.isnull().sum().sum()
    assert initial_missing > 0  # Ensure we have missing values to test

    # Handle missing values
    processed_df = handle_missing_values(sample_data.copy())

    # Check if all missing values were handled
    final_missing = processed_df.isnull().sum().sum()
    assert final_missing == 0

    # Check if numeric columns were filled with median
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if sample_data[col].isnull().sum() > 0:
            assert processed_df[col].equals(
                sample_data[col].fillna(sample_data[col].median())
            )


def test_handle_outliers(sample_data):
    """Test outlier handling"""
    numeric_cols = ["Balance", "Age", "EstimatedSalary"]

    # Process outliers
    processed_df = handle_outliers(sample_data.copy(), numeric_cols)

    # Check if extreme values were capped
    for col in numeric_cols:
        Q1 = sample_data[col].quantile(0.25)
        Q3 = sample_data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        assert processed_df[col].min() >= lower_bound
        assert processed_df[col].max() <= upper_bound


def test_preprocess_data(sample_data):
    """Test the complete preprocessing pipeline"""
    # Run preprocessing
    processed_df, scaler = preprocess_data(sample_data.copy())

    # Check if result is a DataFrame
    assert isinstance(processed_df, pd.DataFrame)
    assert isinstance(scaler, StandardScaler)

    # Check if all missing values were handled
    assert processed_df.isnull().sum().sum() == 0

    # Check if categorical variables were encoded
    assert all(processed_df.dtypes != "object")

    # Check if numeric features were scaled
    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
    numeric_cols = (
        numeric_cols.drop("Churn") if "Churn" in numeric_cols else numeric_cols
    )

    for col in numeric_cols:
        scaled_col = processed_df[col]
        assert abs(scaled_col.mean()) < 1e-5, f"Mean of {col} is not close to 0."
        assert abs(scaled_col.std() - 1.0) < 1e-3, (
            "Standard deviation is not close to 1."
        )


def test_split_data(sample_data):
    """Test data splitting functionality"""
    # Preprocess data first
    processed_df, _ = preprocess_data(sample_data.copy())

    # Test with different test sizes
    test_sizes = [0.2, 0.3]
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = split_data(
            processed_df, target_column="Churn", test_size=test_size
        )

        # Check shapes
        expected_test_size = int(len(processed_df) * test_size)
        assert len(X_test) == expected_test_size
        assert len(y_test) == expected_test_size
        assert len(X_train) == len(processed_df) - expected_test_size
        assert len(y_train) == len(processed_df) - expected_test_size

        # Check if target column is not in features
        assert "Churn" not in X_train.columns
        assert "Churn" not in X_test.columns


def test_split_data_stratification(sample_data):
    """Test stratification in data splitting"""
    # Create imbalanced dataset
    imbalanced_data = sample_data.copy()
    imbalanced_data["Churn"] = np.random.choice(
        [0, 1], size=len(sample_data), p=[0.9, 0.1]
    )

    # Preprocess and split data
    processed_df, _ = preprocess_data(imbalanced_data)
    X_train, X_test, y_train, y_test = split_data(
        processed_df, target_column="Churn", test_size=0.2
    )

    # Check if class proportions are similar in train and test
    train_prop = y_train.mean()
    test_prop = y_test.mean()
    assert abs(train_prop - test_prop) < 0.05  # Allow 5% difference


@pytest.mark.parametrize("test_size", [0.2, 0.3])
def test_split_data_sizes(sample_data, test_size):
    """Test different split sizes"""
    processed_df, _ = preprocess_data(sample_data.copy())
    X_train, X_test, y_train, y_test = split_data(
        processed_df, target_column="Churn", test_size=test_size
    )

    expected_test_size = int(len(processed_df) * test_size)
    assert len(X_test) == expected_test_size
    assert len(y_test) == expected_test_size


def test_data_leakage(sample_data):
    """Test for potential data leakage"""
    # Preprocess data
    processed_df, scaler = preprocess_data(sample_data.copy())

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        processed_df, target_column="Churn", test_size=0.2
    )

    # Check if scaling was done after splitting
    train_mean = X_train.mean()
    test_mean = X_test.mean()

    # Means should be different (but not too different)
    for col in X_train.columns:
        assert abs(train_mean[col] - test_mean[col]) < 0.5


def test_reproducibility(sample_data):
    """Test reproducibility of the preprocessing pipeline"""
    # Run pipeline twice with same random_state
    processed_df1, _ = preprocess_data(sample_data.copy())
    X_train1, X_test1, y_train1, y_test1 = split_data(
        processed_df1, target_column="Churn", random_state=42
    )

    processed_df2, _ = preprocess_data(sample_data.copy())
    X_train2, X_test2, y_train2, y_test2 = split_data(
        processed_df2, target_column="Churn", random_state=42
    )

    # Check if results are identical
    assert X_train1.equals(X_train2)
    assert X_test1.equals(X_test2)
    assert y_train1.equals(y_train2)
    assert y_test1.equals(y_test2)
