import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)


def analyze_data(df):
    """Analyze data and generate basic statistics and visualizations"""

    print("\nDetailed Data Analysis:")
    print("\nBasic Statistics:")
    print(df.describe(include="all"))  # Include all columns
    print("\nMissing Values Analysis:")
    print(df.isnull().sum())
    print("\nMissing Values Percentage:")
    print((df.isnull().sum() / len(df) * 100).round(2))

    # Enhanced visualizations
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig("data/correlation_matrix.png")
    plt.close()

    # Distribution plots for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(15, 5 * ((len(numeric_cols) + 2) // 3)))
    for idx, col in enumerate(numeric_cols, 1):
        plt.subplot((len(numeric_cols) + 2) // 3, 3, idx)
        sns.histplot(df[col], kde=True)
        plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig("data/numeric_distributions.png")
    plt.close()

    return df


def handle_missing_values(df):
    """Handle missing values with appropriate strategies"""
    # Log initial missing values
    print("\nHandling missing values...")
    initial_missing = df.isnull().sum()
    if initial_missing.sum() > 0:
        print("Initial missing values per column:")
        print(initial_missing[initial_missing > 0])

    # Numeric columns: fill with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Filled {col} missing values with median: {median_value:.2f}")

    # Categorical columns: fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Filled {col} missing values with mode: {mode_value}")

    # Verify no missing values remain
    final_missing = df.isnull().sum().sum()
    print(f"\nRemaining missing values: {final_missing}")

    return df


def handle_outliers(df, numeric_cols):
    """Handle outliers using IQR method"""
    print("\nHandling outliers...")
    outlier_stats = {}

    for col in numeric_cols:
        # Calculate statistics
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers before handling
        outliers_lower = (df[col] < lower_bound).sum()
        outliers_upper = (df[col] > upper_bound).sum()
        total_outliers = outliers_lower + outliers_upper

        if total_outliers > 0:
            # Store statistics
            outlier_stats[col] = {
                "total_outliers": total_outliers,
                "lower_outliers": outliers_lower,
                "upper_outliers": outliers_upper,
                "percentage": (total_outliers / len(df) * 100).round(2),
            }

            # Cap outliers to bounds
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Print outlier summary
    if outlier_stats:
        print("\nOutlier Summary:")
        for col, stats in outlier_stats.items():
            print(f"\n{col}:")
            print(
                f"  Total outliers: {stats['total_outliers']} ({stats['percentage']}%)"
            )
            print(f"  Lower bound outliers: {stats['lower_outliers']}")
            print(f"  Upper bound outliers: {stats['upper_outliers']}")
    else:
        print("No outliers found in the dataset.")

    return df


def preprocess_data(df):
    """Main preprocessing pipeline"""
    print("\nStarting preprocessing pipeline...")

    # Analyze data first
    print("\nStep 1: Analyzing data...")
    df = analyze_data(df)

    # Handle missing values
    print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)

    # Handle outliers for numeric columns (excluding target variable)
    print("\nStep 3: Handling outliers...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = (
        numeric_cols.drop("Churn") if "Churn" in numeric_cols else numeric_cols
    )
    df = handle_outliers(df, numeric_cols)

    # Convert categorical variables
    print("\nStep 4: Converting categorical variables...")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if not categorical_cols.empty:
        print(f"Converting columns: {', '.join(categorical_cols)}")
        df = pd.get_dummies(df, drop_first=True)

    # Scale numeric features
    print("\nStep 5: Scaling numeric features...")
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"Scaled {len(numeric_cols)} numeric features")

    print("\nPreprocessing completed!")
    return df, scaler


def split_data(df, target_column, test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print(f"\nSplitting data (test_size={test_size})...")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Check if we have enough samples for stratification
    min_samples_per_class = y.value_counts().min()
    use_stratify = min_samples_per_class >= 2

    if not use_stratify:
        print(
            "Warning: Not enough samples for stratification. Performing regular split."
        )
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(
        f"Target distribution in train: \n{y_train.value_counts(normalize=True).round(3)}"
    )
    print(
        f"Target distribution in test: \n{y_test.value_counts(normalize=True).round(3)}"
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = load_data("data/customer_data.csv")

    # Preprocess data
    print("Preprocessing data...")
    processed_data, scaler = preprocess_data(data)

    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_data, "Churn")

    # Save processed datasets
    print("Saving processed data...")
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("Preprocessing completed successfully!")
