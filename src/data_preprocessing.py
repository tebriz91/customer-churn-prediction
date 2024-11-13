import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    logger.info("Starting detailed data analysis")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Basic statistics and missing values analysis
    logger.info("Basic Statistics:")
    logger.info("\n%s", df.describe(include="all"))
    logger.info("Missing Values Analysis:")
    logger.info("\n%s", df.isnull().sum())
    logger.info("Missing Values Percentage:")
    logger.info("\n%s", df.isnull().sum() * 100 / len(df))

    # Correlation matrix
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df[numeric_cols].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
    plt.close()

    # Scatter plots for most relevant numeric features
    if "Churn" in df.columns:
        # Calculate correlations with target
        correlations = df[numeric_cols].corr()["Churn"].abs()
        # Get top 6 most correlated features (excluding Churn itself)
        top_features = correlations.nlargest(7)[1:7].index

        # Create 2x3 subplot for top 6 features
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(top_features, 1):
            plt.subplot(2, 3, i)
            plt.scatter(
                df[col],
                df["Churn"],
                alpha=0.5,
                c=df["Churn"],
                cmap="coolwarm",
            )
            plt.xlabel(col)
            plt.ylabel("Churn")
            plt.title(f"{col} vs Churn\nCorr: {correlations[col]:.3f}")

        plt.suptitle("Top 6 Features by Correlation with Churn", y=1.02)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "feature_vs_target_scatter.png"),
            bbox_inches="tight",
        )
        plt.close()

    # Distribution plots
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    n_cols = len(numeric_columns)
    n_rows = (n_cols + 2) // 3

    plt.figure(figsize=(15, 5 * n_rows))
    for i, column in enumerate(numeric_columns, 1):
        plt.subplot(n_rows, 3, i)
        sns.histplot(data=df, x=column, kde=True)
        plt.title(f"Distribution of {column}")
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_distributions.png"))
    plt.close()

    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    logger.info("Handling missing values...")

    # Get initial missing values count
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]

    if len(missing_cols) > 0:
        logger.info("Initial missing values per column:")
        logger.info("\n%s", missing_cols)

        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        for col in missing_cols.index:
            if col in numeric_cols:
                median_value = df[col].median()
                df = df.assign(**{col: df[col].fillna(median_value)})
                logger.info(
                    "Filled %s missing values with median: %.2f", col, median_value
                )

        # Categorical columns: fill with mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in missing_cols.index:
            if col in categorical_cols:
                mode_value = df[col].mode()[0]
                df = df.assign(**{col: df[col].fillna(mode_value)})
                logger.info("Filled %s missing values with mode: %s", col, mode_value)

    logger.info("Remaining missing values: %d", df.isnull().sum().sum())
    return df


def handle_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Handle outliers using IQR method"""
    logger.info("Handling outliers...")
    logger.info("Outlier Summary:")

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
        logger.info("\n%s:", col)
        logger.info(
            "  Total outliers: %d (%.1f%%)",
            stats["total_outliers"],
            stats["percentage"],
        )
        logger.info("  Lower bound outliers: %d", stats["lower_outliers"])
        logger.info("  Upper bound outliers: %d", stats["upper_outliers"])

        # Cap outliers
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def plot_raw_data_issues(df: pd.DataFrame, output_dir: str = "data") -> None:
    """Create plot highlighting missing values in raw data."""
    logger.info("Creating plot to highlight missing values...")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    plt.figure(figsize=(10, 6))

    # Plot showing missing values
    missing_data = (df.isnull().sum() / len(df)) * 100
    missing_data.plot(kind="bar")
    plt.title("Missing Values by Feature")
    plt.xlabel("Features")
    plt.ylabel("Missing Values (%)")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "raw_data_issues.png"), bbox_inches="tight", dpi=300
    )
    plt.close()

    # Log summary statistics
    logger.info("Missing values summary:")
    for feature, pct in missing_data.items():
        if pct > 0:
            logger.info(f"  {feature}: {pct:.1f}%")

    logger.info("Raw data quality plot saved to raw_data_issues.png")


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """Preprocess data for modeling"""
    logger.info("Starting preprocessing pipeline...")

    # Make a copy to avoid modifying original data
    df = df.copy()

    # Plot raw data issues before preprocessing
    plot_raw_data_issues(df)

    # Step 1: Analyze data
    logger.info("Step 1: Analyzing data...")
    df = analyze_data(df)

    # Step 2: Handle missing values
    logger.info("Step 2: Handling missing values...")
    df = handle_missing_values(df)

    # Step 3: Handle outliers before scaling
    logger.info("Step 3: Handling outliers...")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    numeric_cols = (
        numeric_cols.drop("Churn") if "Churn" in numeric_cols else numeric_cols
    )
    df = handle_outliers(df, numeric_cols=numeric_cols)

    # Step 4: Convert categorical variables with OneHotEncoder
    logger.info("Step 4: Converting categorical variables...")
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        logger.info("Converting columns: %s", ", ".join(categorical_cols))
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded_data = encoder.fit_transform(df[categorical_cols])
        encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

        df = pd.concat(
            [
                df.drop(categorical_cols, axis=1),
                pd.DataFrame(encoded_data, columns=encoded_feature_names),
            ],
            axis=1,
        )

    # Step 5: Scale numeric features
    logger.info("Step 5: Scaling numeric features...")
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        logger.info("Scaled %d numeric features", len(numeric_cols))
        return df, scaler

    logger.info("Preprocessing completed!")
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

    logger.info("Train set size: %d", len(X_train))
    logger.info("Test set size: %d", len(X_test))
    logger.info(
        "Target distribution in train:\n%s",
        y_train.value_counts(normalize=True).round(3),
    )
    logger.info(
        "Target distribution in test:\n%s", y_test.value_counts(normalize=True).round(3)
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load data
    logger.info("Loading data...")
    data = load_data("data/customer_data.csv")

    # Preprocess data
    logger.info("Preprocessing data...")
    processed_data, scaler = preprocess_data(data)

    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(processed_data, "Churn")

    # Save processed datasets
    logger.info("Saving processed data...")
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    logger.info("Preprocessing completed successfully!")
