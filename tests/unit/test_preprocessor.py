import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    return pd.DataFrame({
        "Age": [25, np.nan, 35],
        "Balance": [1000, 2000, np.nan],
        "Gender": ["M", "F", "M"],
        "Geography": ["USA", "Mexico", "USA"],
    })


def test_handle_missing_values(sample_data):
    """Test missing values handling"""
    # Test numeric imputation
    numeric_cols = ["Age", "Balance"]
    for col in numeric_cols:
        assert sample_data[col].isna().sum() > 0
        filled_data = sample_data[col].fillna(sample_data[col].mean())
        assert filled_data.isna().sum() == 0


def test_validate_schema(sample_data):
    """Test data schema validation"""
    required_columns = ["Age", "Balance", "Gender", "Geography"]
    assert all(col in sample_data.columns for col in required_columns)

    # Test data types
    assert pd.api.types.is_numeric_dtype(sample_data["Age"])
    assert pd.api.types.is_numeric_dtype(sample_data["Balance"])
    assert pd.api.types.is_string_dtype(sample_data["Gender"])
    assert pd.api.types.is_string_dtype(sample_data["Geography"])


def test_encode_categorical_features(sample_data):
    """Test categorical feature encoding"""
    categorical_cols = ["Gender", "Geography"]

    # Test one-hot encoding
    encoded_data = pd.get_dummies(sample_data, columns=categorical_cols)
    expected_cols = [
        "Age",
        "Balance",
        "Gender_F",
        "Gender_M",
        "Geography_Mexico",
        "Geography_USA",
    ]
    assert all(col in encoded_data.columns for col in expected_cols)
