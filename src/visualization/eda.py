import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


def plot_raw_data_issues(df):
    """Plot missing values and other data quality issues."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=True, cmap="viridis")
    plt.title("Missing Value Analysis")
    plt.tight_layout()
    return plt.gcf()
