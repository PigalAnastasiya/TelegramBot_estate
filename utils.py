import re
import nltk
import requests
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from io import BytesIO
from PIL import Image

# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Объединение удаления шума, нормализации и удаления стоп-слов в одну функцию.
def preprocess_text(text, language='english'):
   
    # Удаление шума
    text = re.sub(r'[^\w\s]', '', text)  # Удаление знаков препинания
    text = re.sub(r'_', '', text)        # Удаление нижних подчеркиваний
    text = re.sub(r'\d', '', text)       # Удаление чисел

    # Нормализация текста
    text = text.lower()

    # Удаление стоп-слов
    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text)
    text = [word for word in word_tokens if word not in stop_words]
    
    # Возвращение обработанного текста
    return ' '.join(text)

# Функция замены слов
def preprocess_user_query(query):
    
    # Замена синонимов
    synonyms = {
        "buy": "sell", 
        "Buy": "Sell",
        "to rent":"surrender",
        "To rent":"Surrender",
        "rental":"surrender",
        "Rental":"Surrender"  
    }
    
    for word, synonym in synonyms.items():
        query= re.sub(r'\b' + re.escape(word) + r'\b', synonym, query)

    return query

def enhance_text_with_keywords(text, keywords):
    for word, weight in keywords.items():
        if word in text:
            text += (' ' + word) * (weight - 1)  # увеличиваем количество упоминаний ключевого слова
    return text

# Функция для проверки URL изображения
def is_valid_image_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image.verify()
            return True
        return False
    except Exception as e:
        print(f"Ошибка при проверке URL {url}: {e}")
        return False
    