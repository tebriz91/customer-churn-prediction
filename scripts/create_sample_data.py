"""Script to generate sample customer data for testing."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/feature_config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def create_base_features(n_samples: int) -> pd.DataFrame:
    """Create base numeric and categorical features."""
    data = {
        "CustomerID": range(1, n_samples + 1),
        # Numeric features
        "Age": np.random.randint(18, 80, n_samples),
        "Balance": np.random.uniform(0, 100000, n_samples),
        "EstimatedSalary": np.random.uniform(30000, 120000, n_samples),
        "NumOfProducts": np.random.randint(1, 5, n_samples),
        "Tenure": np.random.randint(0, 20, n_samples),
        "CreditScore": np.random.randint(300, 850, n_samples),
        "MonthlyCharges": np.random.uniform(30, 500, n_samples),
        "TotalCharges": np.random.uniform(1000, 10000, n_samples),
        "NumTransactions": np.random.randint(0, 100, n_samples),
        # Categorical features
        "Gender": np.random.choice(["M", "F"], n_samples),
        "Geography": np.random.choice(["Mexico", "USA"], n_samples),
        "HasCrCard": np.random.choice([0, 1], n_samples),
        "IsActiveMember": np.random.choice([0, 1], n_samples),
    }
    return pd.DataFrame(data)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the DataFrame."""
    # Balance to salary ratio
    df["balance_salary_ratio"] = df["Balance"] / df["EstimatedSalary"]

    # Products per tenure (avoid division by zero)
    df["products_per_tenure"] = df["NumOfProducts"] / df["Tenure"].replace(0, 1)

    # Active customers with credit card
    df["active_with_credit_card"] = (
        (df["IsActiveMember"] == 1) & (df["HasCrCard"] == 1)
    ).astype(int)

    # Age groups
    bins = [0, 25, 35, 45, 55, 100]
    labels = ["18-25", "26-35", "36-45", "46-55", "55+"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    return df


def encode_features(df: pd.DataFrame, features_to_encode: List[str]) -> pd.DataFrame:
    """One-hot encode specified features."""
    # Get dummies for specified features
    df_encoded = pd.get_dummies(df, columns=features_to_encode, drop_first=True)

    # Ensure all expected columns exist
    for feature in features_to_encode:
        if feature == "Geography":
            expected_values = ["Mexico", "USA"]
            for value in expected_values:
                col = f"{feature}_{value}"
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
        elif feature == "Gender":
            col = "Gender_M"  # Only M encoded, F is reference
            if col not in df_encoded.columns:
                df_encoded[col] = 0

    return df_encoded


def main():
    """Generate sample customer data."""
    # Load configuration
    config = load_config()

    # Create sample data
    n_samples = 100
    df = create_base_features(n_samples)

    # Add derived features
    df = add_derived_features(df)

    # Encode categorical features
    features_to_encode = config["data"]["encoding"]["one_hot_encode"]
    df = encode_features(df, features_to_encode)

    # Create output directory if needed
    os.makedirs("data", exist_ok=True)

    # Save to CSV
    output_path = "data/new_customers.csv"
    df.to_csv(output_path, index=False)
    print(f"Created sample data: {output_path}")

    # Print feature summary
    print("\nFeature Summary:")
    print(f"Total features: {len(df.columns)}")
    print("\nFeatures:")
    for i, col in enumerate(df.columns, 1):
        print(f"{i:2d}. {col}")


if __name__ == "__main__":
    main()
