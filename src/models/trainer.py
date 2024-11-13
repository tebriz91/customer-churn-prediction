from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "random_forest": RandomForestClassifier(random_state=random_state),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "logistic_regression": LogisticRegression(random_state=random_state),
        }

    def train_and_evaluate_models(self, X_train, y_train):
        """Train and evaluate multiple models"""
        results = {}
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            results[name] = self._evaluate_model(model, X_train, y_train)
        return results

    def _evaluate_model(self, model, X, y):
        """Evaluate a single model"""
        return {"model": model, "train_score": model.score(X, y)}
