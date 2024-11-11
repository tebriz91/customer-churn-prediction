import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)

    # Гиперпараметрическая настройка
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train_fe.csv")
    y_train = pd.read_csv("data/y_train.csv").values.ravel()

    model = train_model(X_train, y_train)
    joblib.dump(model, "models/churn_model.pkl")
