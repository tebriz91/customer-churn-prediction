import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report


if __name__ == "__main__":
    X_test = pd.read_csv("data/X_test_fe.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    model = joblib.load("models/churn_model.pkl")
    accuracy, report = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(f"Classification Report:\n{report}")
