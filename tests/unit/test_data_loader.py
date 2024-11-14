import pandas as pd
import pytest


def test_load_data(tmp_path):
    """Test data loading functionality"""
    # Create sample data
    sample_data = pd.DataFrame({
        "Age": [25, 30, 35],
        "Balance": [1000, 2000, 3000],
        "Churn": [0, 1, 0],
    })

    # Save sample data
    data_path = tmp_path / "sample_data.csv"
    sample_data.to_csv(data_path, index=False)

    # Test loading
    loaded_data = pd.read_csv(data_path)
    assert loaded_data.shape == (3, 3)
    assert all(col in loaded_data.columns for col in ["Age", "Balance", "Churn"])


def test_load_invalid_data():
    """Test handling of invalid data path"""
    with pytest.raises(FileNotFoundError):
        pd.read_csv("nonexistent_file.csv")


def test_load_empty_data(tmp_path):
    """Test handling of empty data file"""
    # Create empty file
    data_path = tmp_path / "empty.csv"
    data_path.touch()

    with pytest.raises(pd.errors.EmptyDataError):
        pd.read_csv(data_path)
