import pandas as pd
import logging
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel,MarianMTModel, MarianTokenizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем пустой DataFrame с необходимыми колонками
columns = ['Город', 'Общая площадь', 'Этаж', 'Количество спален', 'Количество ванных комнат', 'Количество балконов / террас', 'Дополнительная характеристика', 'Вид из окна / балкона', 'Расстояние до моря', 'Описание', 'Цена', 'Cсылка']
df = pd.DataFrame(columns=columns)

# Настройка Selenium WebDriver
driver = webdriver.Chrome()
#river = webdriver.Chrome(executable_path='./chromedriver.exe', service_log_path='chromedriver.log')


# Перебор страниц
for i in range(913, 1904):
    url = f'https://www.prometheyre.com/ru/details/{i}/'
    driver.get(url)

    # Получение исходного кода страницы
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    description_div = soup.find('div', class_='description-list reveal')
    description_text = description_div.get_text(strip=True) if description_div else 'Описание отсутствует'
    price_div = soup.find('div', class_='item-price')
    price_text = price_div.get_text(strip=True) if price_div else 'Цена не указана'
    left_corner_div = soup.find('div', class_='left-corner')
    status_text = left_corner_div.get_text(strip=True) if left_corner_div else 'Статус не указан'
    property_tables_div = soup.find('div', class_='property-tables reveal')
    property_cont_section = soup.find('section', class_='page-width property-cont')
    images = property_cont_section.find_all('img') if property_cont_section else []
    image_urls = [img['src'] for img in images if 'src' in img.attrs]    
    image_urls_str = '\n'.join(image_urls)
    
    
    status_div = soup.find('div', class_='closed')
    status_text = status_div.get_text(strip=True) if status_div else 'Статус не указан'

    properties = {}
    if property_tables_div:
        for row in property_tables_div.find_all('tr'):
            columns = row.find_all('td')
            if len(columns) == 2:
                key = columns[0].get_text(strip=True)
                value = columns[1].get_text(strip=True)
                properties[key] = value

    properties['Описание'] = description_text
    properties['Цена'] = price_text
    properties['Статус'] = status_text
    properties['Фотографии'] = image_urls_str
    properties['Статус продажи'] = status_text
    properties['Ссылка'] = url
    # data = {извлекаемые данные}

    # Добавление данных в DataFrame
    #print(properties)
    new_row = pd.DataFrame([properties])
    df = pd.concat([df, new_row], ignore_index=True)

driver.quit()
print(df)
df.to_csv('filename.csv', index=False, encoding='utf-8-sig', sep=';')

# Загрузите CSV-файлы и подготовьте данные
df_estate = pd.read_csv('filename.csv', sep=';')
for value in df_estate['Описание']:
    print(value)

df_estate['Описание'] = df_estate['Описание'].fillna('')  # Замена NaN на пустые строки
# Загрузка модели машинного перевода

model_name = 'Helsinki-NLP/opus-mt-ru-en'
tokenizer_translate = MarianTokenizer.from_pretrained(model_name)
model_translate = MarianMTModel.from_pretrained(model_name)

# Функция для перевода текста
def translate(text):
    inputs = tokenizer_translate(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model_translate.generate(**inputs)
    return tokenizer_translate.decode(translated[0], skip_special_tokens=True)


# Перевод столбца 'description'
df_estate['translated_description'] = df_estate['Описание'].apply(translate)


# Загрузка необходимых ресурсов NLTK
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text, language='english'):
    """Объединение удаления шума, нормализации и удаления стоп-слов в одну функцию."""
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



df_estate['processed_text'] = df_estate['translated_description'].apply(preprocess_text)


# Загрузка необходимых ресурсов
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Функция для определения части речи слова
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)




df_estate['lemmatized_text'] = df_estate['translated_description'].apply(
    lambda x: ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(x)])
)

# Загрузка моделей и токенизаторов
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")

tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModel.from_pretrained("roberta-base")

# Функция для создания эмбеддингов
def get_embedding(text):
    inputs = tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def get_embedding_for_model_and_tokenizer(model, tokenizer):
    def get_embedding(text):
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return get_embedding

# Создаем функцию get_embedding специально для model_bert и tokenizer_bert
bert_embedding_function = get_embedding_for_model_and_tokenizer(model_bert, tokenizer_bert)

# Теперь применяем эту функцию к столбцу DataFrame
embedding_bert = df_estate['lemmatized_text'].apply(bert_embedding_function)

# Создаем функцию get_embedding специально для model_bert и tokenizer_bert
roberta_embedding_function = get_embedding_for_model_and_tokenizer(model_roberta, tokenizer_roberta)

# Теперь применяем эту функцию к столбцу DataFrame
embedding_roberta = df_estate['lemmatized_text'].apply(roberta_embedding_function)

df_estate['embedding_ensemble']=(embedding_bert+embedding_roberta)/2


df_estate.to_csv('prod_database_last.csv', index=False,encoding='utf-8-sig', sep=';')

