#Логистическая регрессия и корреляционный отбор признаков
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Загрузка данных
df = pd.read_csv(r'C:\Users\User\OneDrive\Desktop\SVXD/pluginfile.txt', sep='\t', header=0)

# Переименование столбцов для удобства
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df.columns = columns

# Разделение данных на признаки и целевую переменную
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# На основании корреляционного отбора признаков мы убираем столбцы "SkinThickness" и "BloodPressure" из файла данных.

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Добавление столбца единиц к признакам
X_train_scaled = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])
X_test_scaled = np.hstack([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled])

# Определение сигмоидной функции
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Функция потерь
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    cost = (1/m) * ((-y).T @ np.log(h + epsilon) - (1 - y).T @ np.log(1 - h + epsilon))
    return cost

# Градиентный спуск
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        gradient = (1/m) * X.T @ (sigmoid(X @ theta) - y)
        theta = theta - alpha * gradient
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Инициализация параметров
theta_initial = np.zeros(X_train_scaled.shape[1])
alpha = 0.01
iterations = 1000

# Выполнение градиентного спуска
theta_optimized, cost_history = gradient_descent(X_train_scaled, y_train, theta_initial, alpha, iterations)

# Предсказание и оценка точности
y_pred = [1 if x >= 0.5 else 0 for x in sigmoid(X_test_scaled @ theta_optimized)]
accuracy = accuracy_score(y_test, y_pred)

# Отбор признаков на основе корреляции
correlation_with_target = abs(X.corrwith(y))
features_selected = correlation_with_target.sort_values(ascending=False).index[:-2]

# Масштабирование отобранных признаков
X_train_selected = X_train[features_selected]
X_test_selected = X_test[features_selected]
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Добавление столбца единиц и повтор градиентного спуска
X_train_selected_scaled = np.hstack([np.ones((X_train_selected_scaled.shape[0], 1)), X_train_selected_scaled])
X_test_selected_scaled = np.hstack([np.ones((X_test_selected_scaled.shape[0], 1)), X_test_selected_scaled])
theta_initial_selected = np.zeros(X_train_selected_scaled.shape[1])
theta_optimized_selected, cost_history_selected = gradient_descent(X_train_selected_scaled, y_train, theta_initial_selected, alpha, iterations)

# Предсказание и оценка точности с отобранными признаками
y_pred_selected = [1 if x >= 0.5 else 0 for x in sigmoid(X_test_selected_scaled @ theta_optimized_selected)]
accuracy_selected = accuracy_score(y_test, y_pred_selected)

print(f"Точность модели с использованием всех признаков: {accuracy}")
print(f"Точность модели с использованием отобранных признаков: {accuracy_selected}")