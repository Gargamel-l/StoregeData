import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif


###Подготовка данных
# Загрузка данных
df = pd.read_csv(r'C:\programming\WorkZone\CommivoyadgerAlgorithm\ggg\example_data.csv')

# Предположим, что последний столбец — это целевая переменная
X = df.iloc[:, :-1]
y = df.iloc[:, -1]


###Корреляционный отбор признаков
# Отбор K лучших признаков на основе ANOVA F-значения
k = 5  # Количество признаков для отбора
selector = SelectKBest(f_classif, k=k)
X_new = selector.fit_transform(X, y)

# Разбиение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

###Обучение логистической регрессии
# Создание модели логистической регрессии
model = LogisticRegression(max_iter=1000)  # Увеличиваем max_iter при необходимости

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

