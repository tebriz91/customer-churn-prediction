import numpy as np
import pandas as pd
import pytest

from beautifulcode.model_training import ModelTrainer


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.random.randint(0, 2, n_samples))

    return X, y


@pytest.fixture
def trainer():
    """Create a ModelTrainer instance."""
    return ModelTrainer(random_state=42)


def test_model_initialization(trainer):
    """Test if models and param_grids are properly initialized."""
    assert len(trainer.models) == 3
    assert "random_forest" in trainer.models
    assert "gradient_boosting" in trainer.models
    assert "logistic_regression" in trainer.models

    assert len(trainer.param_grids) == 3
    assert all(model in trainer.param_grids for model in trainer.models)


def test_train_and_evaluate_models(trainer, sample_data):
    """Test model training and evaluation."""
    X, y = sample_data
    results = trainer.train_and_evaluate_models(X, y)

    assert len(results) == 3
    for model_name in trainer.models:
        assert model_name in results
        assert "accuracy" in results[model_name]
        assert "precision" in results[model_name]
        assert "recall" in results[model_name]
        assert "f1" in results[model_name]
        assert "std_f1" in results[model_name]

        # Check if metrics are within valid range [0, 1]
        for metric in ["accuracy", "precision", "recall", "f1"]:
            assert 0 <= results[model_name][metric] <= 1


def test_optimize_model(trainer, sample_data):
    """Test model optimization."""
    X, y = sample_data
    model_name = "random_forest"

    best_model, best_params = trainer.optimize_model(X, y, model_name)

    assert best_model is not None
    assert isinstance(best_params, dict)
    assert all(param in trainer.param_grids[model_name] for param in best_params)


def test_feature_importance(trainer, sample_data, tmp_path):
    """Test feature importance calculation and plotting."""
    X, y = sample_data

    # Train a model that supports feature importance
    trainer.optimize_model(X, y, "random_forest")

    # Test feature importance calculation
    assert trainer.feature_importance is not None
    assert len(trainer.feature_importance) == X.shape[1]
    assert all(
        col in trainer.feature_importance.columns for col in ["feature", "importance"]
    )

    # Test plotting
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    trainer.plot_feature_importance(str(output_dir))

    assert (output_dir / "feature_importance.png").exists()


def test_save_model(trainer, sample_data, tmp_path):
    """Test model saving functionality."""
    X, y = sample_data
    model_path = tmp_path / "models" / "test_model.pkl"

    # Train a model first
    trainer.optimize_model(X, y, "random_forest")

    # Test saving
    trainer.save_model(str(model_path))
    assert model_path.exists()


def test_small_dataset_error(trainer):
    """Test error handling for datasets that are too small."""
    X = pd.DataFrame(np.random.randn(3, 5))
    y = pd.Series([0, 1, 0])

    with pytest.raises(ValueError, match="Not enough samples"):
        trainer.train_and_evaluate_models(X, y)


def test_save_model_without_training(trainer):
    """Test error when trying to save an untrained model."""
    with pytest.raises(ValueError, match="No model has been trained yet"):
        trainer.save_model()


def test_plot_feature_importance_without_support(trainer, sample_data):
    """Test feature importance plotting for models without feature importance."""
    X, y = sample_data

    # Train logistic regression which doesn't support feature_importances_
    trainer.optimize_model(X, y, "logistic_regression")

    # Should log a warning and return without error
    trainer.plot_feature_importance()
    assert trainer.feature_importance is None


def test_model_reproducibility(sample_data):
    """Test if models with same random_state produce same results."""
    X, y = sample_data

    trainer1 = ModelTrainer(random_state=42)
    trainer2 = ModelTrainer(random_state=42)

    results1 = trainer1.train_and_evaluate_models(X, y)
    results2 = trainer2.train_and_evaluate_models(X, y)

    # Compare results
    for model_name in results1:
        for metric in results1[model_name]:
            assert np.allclose(
                results1[model_name][metric], results2[model_name][metric]
            )


@pytest.mark.parametrize(
    "model_name", ["random_forest", "gradient_boosting", "logistic_regression"]
)
def test_individual_models(trainer, sample_data, model_name):
    """Test each model individually."""
    X, y = sample_data

    best_model, best_params = trainer.optimize_model(X, y, model_name)

    assert best_model is not None
    assert isinstance(best_params, dict)
    assert all(param in trainer.param_grids[model_name] for param in best_params)


def test_invalid_model_name(trainer, sample_data):
    """Test error handling for invalid model names."""
    X, y = sample_data

    with pytest.raises(KeyError):
        trainer.optimize_model(X, y, "invalid_model")
