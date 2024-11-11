import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Создает новые признаки на основе существующих данных

        Args:
            df: Исходный датафрейм

        Returns:
            DataFrame с новыми признаками
        """
        logger.info("Начало создания новых признаков")

        df = df.copy()

        # Создаем признак отношения баланса к зарплате
        df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

        # Создаем признак среднего продукта на год обслуживания
        df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

        # Создаем признак взаимодействия между активностью и кредитной картой
        df["active_with_credit_card"] = df["IsActiveMember"] * df["HasCrCard"]

        # Создаем возрастные группы
        df["age_group"] = pd.qcut(
            df["Age"], q=4, labels=["Young", "Adult", "Middle", "Senior"]
        )

        # One-hot encoding для категориальных признаков
        categorical_columns = ["Gender", "age_group"]
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

        logger.info(
            f"Созданы новые признаки: {list(set(df.columns) - set(df.columns))}"
        )
        return df

    def scale_features(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Масштабирует числовые признаки с помощью StandardScaler

        Args:
            train_df: Тренировочный датасет
            test_df: Тестовый датасет (опционально)

        Returns:
            Кортеж из масштабированных тренировочного и тестового датасетов
        """
        logger.info("Начало масштабирования признаков")

        # Определяем числовые колонки для масштабирования
        numeric_columns = train_df.select_dtypes(include=["float64", "int64"]).columns

        # Масштабируем тренировочные данные
        train_scaled = train_df.copy()
        train_scaled[numeric_columns] = self.scaler.fit_transform(
            train_df[numeric_columns]
        )

        # Масштабируем тестовые данные, если они предоставлены
        test_scaled = None
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[numeric_columns] = self.scaler.transform(
                test_df[numeric_columns]
            )

        logger.info(f"Масштабированы признаки: {list(numeric_columns)}")
        return train_scaled, test_scaled


def main():
    """
    Основная функция для запуска процесса создания признаков
    """
    try:
        # Загружаем данные
        train_data = pd.read_csv("data/X_train.csv")
        test_data = pd.read_csv("data/X_test.csv")

        # Создаем экземпляр класса
        fe = FeatureEngineer()

        # Создаем новые признаки
        train_featured = fe.create_features(train_data)
        test_featured = fe.create_features(test_data)

        # Масштабируем признаки
        train_scaled, test_scaled = fe.scale_features(train_featured, test_featured)

        # Сохраняем обработанные данные
        train_scaled.to_csv("data/X_train_featured.csv", index=False)
        test_scaled.to_csv("data/X_test_featured.csv", index=False)

        logger.info("Процесс создания признаков успешно завершен")

    except Exception as e:
        logger.error(f"Произошла ошибка при создании признаков: {str(e)}")
        raise


if __name__ == "__main__":
    main()
