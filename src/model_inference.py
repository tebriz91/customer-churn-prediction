import joblib
import pandas as pd


def load_model(model_path):
    return joblib.load(model_path)


def predict_churn(model, new_data):
    predictions = model.predict(new_data)
    return predictions


if __name__ == "__main__":
    model = load_model("models/churn_model.pkl")
    new_data = pd.read_csv("data/new_customer_data.csv")

    predictions = predict_churn(model, new_data)
    new_data["ChurnPrediction"] = predictions

    new_data.to_csv("data/new_customer_predictions.csv", index=False)
