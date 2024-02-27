from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Шаг 1: Подготовка данных
texts = [
    "Win a new car! Call now!",
    "Meet singles in your area",
    "Cheap mortgage rates available",
    "Congratulations, you've won a prize!",
    "This is a normal message, not spam",
    "Another regular message",
    "Yet another normal message",
    "This is not spam, just a regular update"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 - спам, 0 - не спам

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.25, random_state=42)

# Шаг 2: Предварительная обработка текста
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Шаг 3: Обучение наивного байесовского классификатора
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Шаг 4: Оценка модели
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Шаг 5: Использование модели для предсказания новых сообщений
new_texts = [
    "Congratulations, you've been selected!",
    "This is a reminder for your meeting tomorrow",
    "Don't miss out on this offer to win big!",
    "Your regular newsletter update"
]
new_texts_vec = vectorizer.transform(new_texts)
predictions = clf.predict(new_texts_vec)

for text, label in zip(new_texts, predictions):
    print(f"Message: \"{text}\" is {'spam' if label == 1 else 'not spam'}")

