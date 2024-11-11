import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Error loading data: {str(e)}")


def analyze_data(df: pd.DataFrame, output_dir: str = "data") -> pd.DataFrame:
    """Analyze data and generate visualizations"""
    logging.info("Starting detailed data analysis")

    # Basic statistics
    logging.info("Basic Statistics:")
    logging.info("\n%s", df.describe(include="all"))
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


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    logging.info("Handling missing values...")

    # Get initial missing values count
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if len(missing_cols) > 0:
        logging.info("Initial missing values per column:")
        logging.info("\n%s", missing_cols)

        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in missing_cols.index:
            if col in numeric_cols:
                median_value = df[col].median()
                df = df.assign(**{col: df[col].fillna(median_value)})
                logging.info(
                    "Filled %s missing values with median: %.2f", col, median_value
                )

        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in missing_cols.index:
            if col in categorical_cols:
                mode_value = df[col].mode()[0]
                df = df.assign(**{col: df[col].fillna(mode_value)})
                logging.info("Filled %s missing values with mode: %s", col, mode_value)

    logging.info("Remaining missing values: %d", df.isnull().sum().sum())
    return df


def handle_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Handle outliers using IQR method"""
    logging.info("Handling outliers...")
    logging.info("Outlier Summary:")

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

        # Log outlier statistics
        logging.info("\n%s:", col)
        logging.info(
            "  Total outliers: %d (%.1f%%)",
            stats["total_outliers"],
            stats["percentage"],
        )
        logging.info("  Lower bound outliers: %d", stats["lower_outliers"])
        logging.info("  Upper bound outliers: %d", stats["upper_outliers"])

        # Cap outliers
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Preprocess data for modeling"""
    logging.info("Starting preprocessing pipeline...")

    # Make a copy to avoid modifying original data
    df = df.copy()

    # Step 1: Analyze data
    logging.info("Step 1: Analyzing data...")
    df = analyze_data(df)

    # Step 2: Handle missing values
    logging.info("Step 2: Handling missing values...")
    df = handle_missing_values(df)

    # Step 3: Handle outliers before scaling
    logging.info("Step 3: Handling outliers...")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = (
        numeric_cols.drop("Churn") if "Churn" in numeric_cols else numeric_cols
    )
    df = handle_outliers(df, numeric_cols=numeric_cols)

    # Step 4: Convert categorical variables
    logging.info("Step 4: Converting categorical variables...")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        logging.info("Converting columns: %s", ", ".join(categorical_cols))
        for col in categorical_cols:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

    # Step 5: Scale numeric features
    logging.info("Step 5: Scaling numeric features...")
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logging.info("Scaled %d numeric features", len(numeric_cols))
        return df, scaler

    logging.info("Preprocessing completed!")
    return df, None


def split_data(
    df: pd.DataFrame,
    target_column: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets"""
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Check if stratification is possible
    min_samples = y.value_counts().min()
    use_stratify = min_samples >= 2

    if not use_stratify:
        logging.warning("Not enough samples for stratification. Using regular split.")
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

    logging.info("Train set size: %d", len(X_train))
    logging.info("Test set size: %d", len(X_test))
    logging.info(
        "Target distribution in train:\n%s",
        y_train.value_counts(normalize=True).round(3),
    )
    logging.info(
        "Target distribution in test:\n%s", y_test.value_counts(normalize=True).round(3)
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load data
    logging.info("Loading data...")
    data = load_data("data/customer_data.csv")

    # Preprocess data
    logging.info("Preprocessing data...")
    processed_data, scaler = preprocess_data(data)

    # Split data
    logging.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_data, "Churn")

    # Save processed datasets
    logging.info("Saving processed data...")
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    logging.info("Preprocessing completed successfully!")
