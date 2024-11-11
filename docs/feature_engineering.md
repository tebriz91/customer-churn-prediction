# Feature Engineering

## Обзор

Этот документ описывает процесс создания и преобразования признаков (feature 
engineering) для модели прогнозирования оттока клиентов банка.

## Созданные признаки

### 1. Отношение баланса к зарплате (balance_salary_ratio)

- **Описание**: Отношение текущего баланса клиента к его предполагаемой зарплате
- **Формула**: `Balance / EstimatedSalary.replace(0, 1)`
- **Обоснование**: Этот признак помогает определить, какую часть своего дохода 
клиент хранит в банке. Низкое значение может указывать на то, что клиент 
предпочитает хранить деньги в другом месте.

### 2. Продукты на год обслуживания (products_per_tenure)

- **Описание**: Среднее количество банковских продуктов на год обслуживания
- **Формула**: `NumOfProducts / (Tenure.replace(0, 1) + 1)`
- **Обоснование**: Показывает интенсивность использования банковских продуктов. 
Высокое значение может указывать на более тесную связь клиента с банком.

### 3. Активность с кредитной картой (active_with_credit_card)

- **Описание**: Взаимодействие между статусом активного клиента и наличием 
кредитной карты
- **Формула**: `IsActiveMember * HasCrCard`
- **Обоснование**: Комбинированный признак, показывающий активных клиентов с 
кредитными картами.

### 4. Возрастные группы (age_group)

- **Описание**: Категоризация клиентов по возрастным группам
- **Метод**: Квартили по возрасту (pd.qcut) без явного указания меток
- **Обоснование**: Разные возрастные группы могут иметь различные паттерны поведения.

## Преобразование признаков

### Кодирование категориальных признаков

- **Метод**: One-hot encoding с drop_first=True
- **Применяется к**:
  - Gender
  - age_group

### Масштабирование числовых признаков

- **Метод**: StandardScaler
- **Применяется ко всем числовым признакам**
- **Особенности**: 
  - Обучается только на тренировочных данных
  - Применяется к тестовым данным без повторного обучения

### Отбор признаков

- **Метод**: SelectKBest с использованием `f_classif`
- **Описание**: Выбирает `k` лучших признаков на основе статистического теста `f_classif`.
- **Аргументы**:
  - `df`: Датафрейм с признаками.
  - `target`: Целевой столбец.
  - `k`: Количество признаков для выбора (по умолчанию 10).
- **Использование**:
  - Применяется после создания и масштабирования признаков для отбора наиболее информативных признаков.

## Использование

### Базовое использование

```python
from beautifulcode.feature_engineering import FeatureEngineer

# Создание экземпляра класса
fe = FeatureEngineer()

# Создание новых признаков
featured_data = fe.create_features(input_data)

# Масштабирование признаков
train_scaled, test_scaled = fe.scale_features(train_featured, test_featured)

# Отбор лучших признаков
train_selected = fe.select_features(train_scaled, target='Exited', k=10)
test_selected = fe.select_features(test_scaled, target='Exited', k=10)
```

### Запуск из командной строки

```bash
python -m beautifulcode.feature_engineering
```

## Входные данные

### Ожидаемые колонки

- `Balance`: Баланс счета
- `EstimatedSalary`: Предполагаемая зарплата
- `NumOfProducts`: Количество продуктов
- `Tenure`: Срок обслуживания
- `IsActiveMember`: Признак активности клиента
- `HasCrCard`: Наличие кредитной карты
- `Age`: Возраст клиента
- `Gender`: Пол клиента

### Форматы файлов

- Входные файлы: CSV
- Выходные файлы: CSV
- Кодировка: UTF-8

## Выходные данные

### Новые признаки

- `balance_salary_ratio`: Отношение баланса к зарплате
- `products_per_tenure`: Продукты на год обслуживания
- `active_with_credit_card`: Активность с кредитной картой
- `age_group`: Возрастная группа
- One-hot encoded признаки для Gender и age_group

### Расположение файлов

- `data/X_train_featured.csv`: Обработанный тренировочный набор
- `data/X_test_featured.csv`: Обработанный тестовый набор

## Логирование

### События

- Начало создания признаков
- Список созданных признаков
- Начало масштабирования
- Список масштабированных признаков
- Начало отбора признаков
- Список выбранных признаков
- Ошибки при обработке

### Уровни

- INFO: Основные этапы обработки
- ERROR: Ошибки при обработке данных

## Зависимости

- pandas
- scikit-learn
- logging (стандартная библиотека)
- typing (стандартная библиотека)

## Обработка ошибок

- Проверка существования входных файлов
- Обработка отсутствующих колонок
- Логирование всех ошибок с подробными сообщениями