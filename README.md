# Хакатон "Beautiful Code"

## Структура проекта

1. `data_preprocessing.py` - скрипт для предобработки данных
2. `feature_engineering.py` - скрипт для создания и отбора признаков
3. `model_training.py` - скрипт для обучения модели машинного обучения
4. `model_evaluation.py` - скрипт для оценки и тестирования модели
5. `model_inference.py` - скрипт для интеграции модели и предсказания оттока клиентов на новых данных
6. `data/` - папка, содержащая обучающие и тестовые данные (например, в формате CSV)

## Описание задачи

**Задача:** Построить и обучить модель машинного обучения для прогнозирования оттока клиентов на основе предоставленного набора исторических данных. Задача включает в себя этапы предобработки данных, выбора и тренировки модели, а также оценки ее точности. Дополнительно необходимо визуализировать результаты и объяснить значимость ключевых факторов, влияющих на отток.

**Цель:** Проверить навыки Data Science и Data Engineering в предобработке данных, создании и обучении модели, в её интеграции и развертывании, а также в обработке и визуализации данных.

## Этапы выполнения

### 1. Предобработка данных

- Осуществите очистку данных, включая удаление пропусков и преобразование категориальных признаков
- Разделите данные на обучающую и тестовую выборки

- Очистить данные от пропусков и аномальных значений.
- Провести анализ данных для выявления корреляций между признаками.
- Преобразовать категориальные переменные в числовые, если это необходимо.
- Разделить данные на тренировочную и тестовую выборки.

### 2. Feature Engineering

- Создайте новые признаки на основе имеющихся данных
- Произведите масштабирование признаков для улучшения качества модели

### 3. Обучение модели

- Обучите модель машинного обучения (например, RandomForest) на подготовленных данных
- Проведите гиперпараметрическую настройку модели

### 4. Оценка модели

- Оцените модель с использованием метрик точности и отчёта по классификации
- Проверьте модель на тестовых данных и проанализируйте результаты

### 5. Интеграция модели

- Реализуйте предсказание оттока для новых данных с использованием обученной модели
- Интегрируйте модель в систему, обеспечив возможность её использования на новых данных

## Требования к исходным данным

**Исходные данные:**

- Набор данных представляет собой таблицу с исторической информацией о клиентах банка.
- Пример структуры данных:
  - CustomerID: Уникальный идентификатор клиента.
  - Age: Возраст клиента.
  - Gender: Пол клиента.
  - Tenure: Срок обслуживания клиента в банке (в годах).
  - Balance: Баланс на счету клиента.
  - NumOfProducts: Количество продуктов банка, используемых клиентом.
  - HasCrCard: Наличие кредитной карты (1 – да, 0 – нет).
  - IsActiveMember: Является ли клиент активным пользователем услуг банка (1 – да, 0 – нет).
  - EstimatedSalary: Оценка заработной платы клиента.
  - Exited: Целевой столбец, который указывает на то, покинул ли клиент банк (1 – да, 0 – нет).

**Дополнительные данные:**

- В наборе данных могут присутствовать дополнительные признаки (features), такие как географическое положение клиента, информация о прошлых транзакциях, кредитный рейтинг и т.д.
- Данные должны содержать пропуски, аномалии или некорректные значения, которые участники должны корректно обработать.

## Задачи

**1. Предобработка данных:**

- Очистить данные от пропусков и аномальных значений.
- Провести анализ данных для выявления корреляций между признаками.
- Преобразовать категориальные переменные в числовые, если это необходимо.
- Разделить данные на тренировочную и тестовую выборки.

**2. Построение модели:**

- Выбрать подходящую модель машинного обучения для задачи прогнозирования оттока (например, логистическая регрессия, случайный лес, градиентный бустинг и т.д.).
- Обучить модель на тренировочной выборке.
- Провести гиперпараметрическую настройку модели для повышения её точности.
- Оценить производительность модели на тестовой выборке.

**3. Оценка и интерпретация модели:**

- Оценить модель с использованием различных метрик, таких как Accuracy, Precision, Recall, F1-score и ROC-AUC.
- Визуализировать важность признаков (feature importance), влияющих на решение модели.
- Подготовить краткий отчет с объяснением ключевых факторов, которые наиболее сильно влияют на отток клиентов.

**4. Визуализация данных:**

- Построить графики, иллюстрирующие распределение данных, корреляции между признаками, а также производительность модели.
- Визуализировать кривую ROC и другие важные метрики для наглядной оценки модели.

## Технические требования

Язык программирования и инструменты:

- Язык программирования: Python.
- Библиотеки для анализа данных: Pandas, NumPy.
- Библиотеки для машинного обучения: Scikit-learn, XGBoost, LightGBM (или другие на выбор).
- Библиотеки для визуализации: Matplotlib, Seaborn, Plotly (или другие на выбор).

## Требования к коду

- Код должен быть хорошо структурированным, с комментариями и разделением на логические блоки.
- Следует использовать лучшие практики в области анализа данных и машинного обучения.
- Обязательно обеспечить воспроизводимость результатов (фиксация seed для случайных процессов, детальная инструкция по запуску кода).

## Инструкция по выполнению

**1. Подготовка данных:**

- Участник должен загрузить предоставленный набор данных и провести его анализ.
- Очистить и предобработать данные, подготовив их для обучения модели.

**2. Моделирование:**

- Выбрать и обучить модель на тренировочных данных.
- Настроить гиперпараметры и оценить модель на тестовой выборке.
- Провести интерпретацию модели и выделить важные признаки.

**3. Визуализация и отчет:**

- Построить графики, иллюстрирующие ключевые моменты анализа и результаты моделирования.
- Подготовить краткий отчет, который включает описание процесса, результаты и выводы.

**4. Проверка и сдача работы:**

- Проверить, что все этапы выполнены корректно и результаты удовлетворяют требованиям задачи.
- Подготовить проект к сдаче, оформив все необходимые документы и инструкции по запуску кода.

## Критерии оценки

### Предобработка и анализ данных

- Очистка данных и работа с пропусками: Оценивается, насколько эффективно участник справился с предобработкой данных. Это включает в себя очистку данных от шумов, работу с пропущенными значениями (заполнение, удаление или замена), а также обработку аномалий. Участник должен продемонстрировать обоснованный подход к этим задачам, учитывая влияние принятых решений на дальнейшую модель.

- Анализ данных и создание признаков: Оценивается качество проведенного анализа данных, включая выявление важных корреляций и зависимостей между признаками. Участник должен продемонстрировать умение правильно интерпретировать данные, а также создать новые информативные признаки (feature engineering), которые могут улучшить качество модели.

- Преобразование категориальных признаков: Проверяется корректность обработки категориальных данных (например, с использованием one-hot encoding, label encoding и других методов). Важно, чтобы выбранный метод был обоснован и соответствовал типу задачи.

- Масштабирование и нормализация данных: Оценивается использование техник масштабирования и нормализации данных для улучшения работы модели. Участник должен понимать, когда и почему необходимо применять данные методы.
Построение и обучение модели

- Выбор модели и обоснование: Оценивается выбор модели машинного обучения для решения задачи. Участник должен обосновать свой выбор, учитывая природу данных, требования к точности и производительности модели. Важно, чтобы выбранная модель соответствовала поставленной задаче и использовала доступные данные максимально эффективно.

- Гиперпараметрическая настройка: Проверяется, насколько тщательно участник подходил к настройке гиперпараметров модели. Оценивается использование подходов, таких как Grid Search, Random Search или использование методов автоматической оптимизации гиперпараметров. Важно, чтобы участник мог улучшить модель, используя правильную настройку параметров.

- Оценка и валидация модели: Оценивается корректность и полнота оценки модели с использованием метрик, таких как accuracy, precision, recall, F1-score, ROC-AUC и других. Важно, чтобы участник использовал подходящие метрики для задачи и правильно интерпретировал результаты. Также оценивается использование кросс-валидации для более точной оценки модели.

- Интерпретация модели и объяснимость: Оценивается, насколько участник понимает и может объяснить работу своей модели. Это включает в себя анализ важности признаков (feature importance), построение SHAP или LIME графиков и интерпретацию влияния признаков на предсказания модели. Важно, чтобы участник мог объяснить, почему модель принимает те или иные решения.

### Инженерия данных (Data Engineering)

- Эффективность работы с большими данными: Оценивается способность участника работать с большими наборами данных, включая оптимизацию операций по загрузке, обработке и сохранению данных. Это может включать использование эффективных алгоритмов, распределенных вычислений или оптимизированных структур данных.

- Стабильность и масштабируемость: Оценивается, насколько разработанное решение устойчиво к масштабированию. Участник должен продемонстрировать, что его подход будет работать эффективно при увеличении объема данных, и предложить решения для масштабирования (например, использование распределенных систем хранения данных или параллельных вычислений).

- Потоковая обработка данных: Если задача предусматривает потоковую обработку данных, оценивается способность участника настроить и реализовать пайплайн для обработки данных в реальном времени. Важно, чтобы участник продемонстрировал понимание архитектуры и принципов работы потоковых систем (например, Apache Kafka, Apache Flink, Spark Streaming).

- Интеграция и автоматизация: Оценивается, насколько хорошо участник интегрировал решение с внешними источниками данных и настроил автоматизацию процессов (например, периодическую загрузку данных, автоматическую предобработку и обновление модели). Важно, чтобы решение было автономным и не требовало постоянного вмешательства.
Визуализация данных и результатов

- Визуализация данных: Оценивается качество визуализации данных, использованных на этапе анализа и предобработки. Участник должен уметь наглядно представить распределение признаков, выявленные зависимости и аномалии. Важно, чтобы визуализации были информативными и помогали в понимании данных.

- Визуализация результатов модели: Проверяется качество визуализации результатов работы модели, включая ROC-кривую, PR-кривую, графики важности признаков и другие визуализации, которые помогают интерпретировать результаты работы модели.

- Интерактивные дашборды (опционально): Если задача предусматривает создание интерактивных дашбордов для мониторинга работы модели или отображения результатов, оценивается их качество, удобство использования и полезность для конечных пользователей.

### Тестирование и воспроизводимость

- Тестирование модели: Оценивается, насколько тщательно участник протестировал свою модель. Важно, чтобы тестирование охватывало различные сценарии, включая пограничные случаи и работу с новыми данными, которые не участвовали в обучении.

- Репродуктивность экспериментов: Оценивается, насколько решение участника воспроизводимо. Важно, чтобы любой другой специалист мог воспроизвести результаты, используя предоставленный код и данные. Участник должен обеспечить фиксацию случайных состояний (seeds) и документирование всех шагов эксперимента.

- Документация и инструкции: Проверяется наличие и качество документации, которая должна включать описание этапов выполнения задания, объяснение выбора методов, использование библиотек и фреймворков, а также инструкции по запуску кода и воспроизведению результатов.

### Рекомендации по созданию тестов

- Тестирование модели: Разделение данных на тренировочную и тестовую выборки должно быть проведено корректно, чтобы избежать утечек данных.

- Кросс-валидация: Использовать кросс-валидацию для более точной оценки модели.

- Анализ ошибок: Провести анализ ошибок модели, чтобы понять, где она работает плохо и почему.