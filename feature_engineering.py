import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_engineering(df):
    # Создание новых признаков (пример)
    df["TotalTransactions"] = df["NumTransactions"] * df["Tenure"]

    # Масштабирование признаков
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    return df


if __name__ == "__main__":
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")

    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    X_train.to_csv("data/X_train_fe.csv", index=False)
    X_test.to_csv("data/X_test_fe.csv", index=False)
