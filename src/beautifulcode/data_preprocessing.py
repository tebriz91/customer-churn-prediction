import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """Load data from CSV file"""
    return pd.read_csv(file_path)


def analyze_data(df, output_dir="data"):
    """Analyze data and generate visualizations"""
    print("\nDetailed Data Analysis:\n")

    # Basic statistics
    print("Basic Statistics:")
    print(df.describe(include="all"))
    print()

    # Missing values analysis
    print("Missing Values Analysis:")
    print(df.isnull().sum())
    print()

    print("Missing Values Percentage:")
    print(df.isnull().sum() * 100 / len(df))
    print()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Correlation matrix for numeric columns only
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # Distribution plots for numeric features
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 3, i)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"{col} Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_distributions.png"))
    plt.close()

    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    print("\nHandling missing values...")

    # Get initial missing values count
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if len(missing_cols) > 0:
        print("Initial missing values per column:")
        print(missing_cols)

        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in missing_cols.index:
            if col in numeric_cols:
                median_value = df[col].median()
                # Fix the FutureWarning by avoiding chained assignment
                df = df.assign(**{col: df[col].fillna(median_value)})
                print(f"Filled {col} missing values with median: {median_value:.2f}")

        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in missing_cols.index:
            if col in categorical_cols:
                mode_value = df[col].mode()[0]
                df = df.assign(**{col: df[col].fillna(mode_value)})
                print(f"Filled {col} missing values with mode: {mode_value}")

    print(f"\nRemaining missing values: {df.isnull().sum().sum()}")
    return df


def handle_outliers(df, numeric_cols):
    """Handle outliers using IQR method"""
    print("\nHandling outliers...")
    print("\nOutlier Summary:")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count outliers
        lower_outliers = (df[col] < lower_bound).sum()
        upper_outliers = (df[col] > upper_bound).sum()
        total_outliers = lower_outliers + upper_outliers
        percentage = round(total_outliers / len(df) * 100, 1)

        stats = {
            "total_outliers": total_outliers,
            "percentage": percentage,
            "lower_outliers": lower_outliers,
            "upper_outliers": upper_outliers,
        }

        # Print outlier statistics
        print(f"\n{col}:")
        print(f"  Total outliers: {stats['total_outliers']} ({stats['percentage']}%)")
        print(f"  Lower bound outliers: {stats['lower_outliers']}")
        print(f"  Upper bound outliers: {stats['upper_outliers']}")

        # Cap outliers
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def preprocess_data(df):
    """Preprocess data for modeling"""
    print("\nStarting preprocessing pipeline...")

    # Make a copy to avoid modifying original data
    df = df.copy()

    # Step 1: Analyze data
    print("\nStep 1: Analyzing data...")
    df = analyze_data(df)

    # Step 2: Handle missing values
    print("\nStep 2: Handling missing values...")
    df = handle_missing_values(df)

    # Step 3: Handle outliers before scaling
    print("\nStep 3: Handling outliers...")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = (
        numeric_cols.drop("Churn") if "Churn" in numeric_cols else numeric_cols
    )
    df = handle_outliers(df, numeric_cols=numeric_cols)

    # Step 4: Convert categorical variables
    print("\nStep 4: Converting categorical variables...")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        print(f"Converting columns: {', '.join(categorical_cols)}")
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    # Step 5: Scale numeric features
    print("\nStep 5: Scaling numeric features...")
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        print(f"Scaled {len(numeric_cols)} numeric features")
        return df, scaler

    print("\nPreprocessing completed!")
    return df, None


def split_data(df, target_column="Churn", test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Check if stratification is possible
    min_samples = y.value_counts().min()
    use_stratify = min_samples >= 2

    if not use_stratify:
        print("Warning: Not enough samples for stratification. Using regular split.")
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(
        "Target distribution in train:\n"
        f"{y_train.value_counts(normalize=True).round(3)}"
    )
    print(
        f"Target distribution in test:\n{y_test.value_counts(normalize=True).round(3)}"
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
