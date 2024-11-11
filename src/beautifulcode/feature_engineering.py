import logging
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
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
        logger.info(f"Доступные колонки: {df.columns.tolist()}")

        df = df.copy()

        # Проверяем наличие необходимых колонок
        required_columns = [
            "Balance",
            "EstimatedSalary",
            "NumOfProducts",
            "Tenure",
            "IsActiveMember",
            "HasCrCard",
            "Age",
            "Gender",
        ]

        # Проверяем какие колонки отсутствуют
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            # Создаем базовые колонки со значениями по умолчанию
            for col in missing_columns:
                if col in ["Balance", "EstimatedSalary", "Age"]:
                    df[col] = 0.0
                elif col in ["NumOfProducts", "Tenure"]:
                    df[col] = 1
                elif col in ["IsActiveMember", "HasCrCard"]:
                    df[col] = 0
                elif col == "Gender":
                    df[col] = "Unknown"
            logger.warning(f"Созданы колонки по умолчанию для: {missing_columns}")

        # Создаем признаки с защитой от деления на ноль
        df["balance_salary_ratio"] = df["Balance"] / (
            df["EstimatedSalary"].replace(0, 1)
        )

        df["products_per_tenure"] = df["NumOfProducts"] / (
            df["Tenure"].replace(0, 1) + 1
        )

        # Создаем признак взаимодействия
        df["active_with_credit_card"] = df["IsActiveMember"] * df["HasCrCard"]

        # Создаем возрастные группы
        df["age_group"] = pd.qcut(
            df["Age"],
            q=4,
            duplicates="drop",
        )

        # One-hot encoding для категориальных признаков
        cat_columns = ["Gender", "age_group"]
        df = pd.get_dummies(df, columns=cat_columns, drop_first=True)

        # Получаем список новых признаков
        new_features = [col for col in df.columns if col not in set(required_columns)]
        logger.info(f"Созданы новые признаки: {new_features}")

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
            Кортеж (масштабированный train, масштабированный test)
        """
        logger.info("Начало масштабирования признаков")

        # Определяем числовые колонки для масштабирования
        numeric_cols = train_df.select_dtypes(include=["float64", "int64"]).columns

        # Масштабируем тренировочные данные
        train_scaled = train_df.copy()
        train_scaled[numeric_cols] = self.scaler.fit_transform(train_df[numeric_cols])

        # Масштабируем тестовые данные, если они предоставлены
        test_scaled = None
        if test_df is not None:
            test_scaled = test_df.copy()
            test_scaled[numeric_cols] = self.scaler.transform(test_df[numeric_cols])

        logger.info(f"Масштабированы признаки: {list(numeric_cols)}")
        return train_scaled, test_scaled

    def select_features(
        self, df: pd.DataFrame, target: str, k: int = 10
    ) -> pd.DataFrame:
        """
        Выбирает лучшие признаки на основе метода SelectKBest

        Args:
            df: Исходный датафрейм с признаками
            target: Целевой столбец
            k: Количество признаков для выбора

        Returns:
            DataFrame с выбранными признаками
        """
        logger.info("Начало отбора признаков с использованием SelectKBest")
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(df, target)
        selected_columns = df.columns[selector.get_support()]
        logger.info(f"Выбраны признаки: {selected_columns.tolist()}")
        return df[selected_columns]


def main():
    """
    Основная функция для запуска процесса создания признаков
    """
    try:
        # Загружаем данные
        train_data = pd.read_csv("data/X_train.csv")
        test_data = pd.read_csv("data/X_test.csv")

        # Выводим информацию о загруженных данных
        logger.info("Колонки в тренировочном наборе:")
        logger.info(train_data.columns.tolist())
        logger.info(f"Размер тренировочного набора: {train_data.shape}")

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
