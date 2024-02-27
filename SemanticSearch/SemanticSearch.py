from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Инициализация модели и токенизатора
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Функция для получения embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Пример текстов
texts = [
    "The cat sits on the mat",
    "A man is playing a guitar",
    "I love studying machine learning"
]

# Вычисление embeddings для текстов
embeddings = [get_embedding(text) for text in texts]

# Пример запроса
query = "musical instruments"
query_embedding = get_embedding(query)

# Оценка близости
similarities = [cosine_similarity(query_embedding, embedding)[0][0] for embedding in embeddings]

# Вывод результатов
for text, similarity in zip(texts, similarities):
    print(f"Text: \"{text}\", Similarity: {similarity}")

