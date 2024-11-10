import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def preprocess_data(df):
    # Удаление пропусков
    df = df.dropna()

    # Преобразование категориальных признаков
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    data = load_data("data/customer_data.csv")
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, "churn")
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
