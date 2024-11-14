from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Model training configuration."""

    name: str = "random_forest"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """Data processing configuration."""

    numeric_features: List[str] = field(
        default_factory=lambda: [
            "Age",
            "Balance",
            "EstimatedSalary",
            "NumOfProducts",
            "Tenure",
        ]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: [
            "Gender",
            "Geography",
            "HasCrCard",
            "IsActiveMember",
        ]
    )
    target_column: str = "Churn"
    derived_features: Optional[List[str]] = field(default_factory=list)


@dataclass
class PathConfig:
    """Path configuration."""

    data_dir: Path = field(default_factory=lambda: Path("data"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def __post_init__(self):
        # Create directories if they don't exist
        for directory in [self.data_dir, self.models_dir, self.output_dir]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file and return Config object."""
    try:
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Convert dictionary paths to Path objects
        if "paths" in config_dict:
            for key, value in config_dict["paths"].items():
                config_dict["paths"][key] = Path(value)

        # Create Config object with loaded values
        config = Config(
            model=ModelConfig(**config_dict.get("model", {})),
            data=DataConfig(**config_dict.get("data", {})),
            paths=PathConfig(**config_dict.get("paths", {})),
        )

        logger.info("Configuration loaded successfully")
        return config

    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        raise
