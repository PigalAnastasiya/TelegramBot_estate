import logging
import numpy as np 
import pandas as pd 
import torch  
import nltk
import requests
from telegram import InputMediaPhoto
from io import BytesIO
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sentence_transformers import util
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import AutoTokenizer, AutoModel
from utils import is_valid_image_url,preprocess_user_query,enhance_text_with_keywords
from nlp import get_wordnet_pos, get_embedding, translate

# Настройка логирования
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# Загрузка данных
df_estate = pd.read_csv('prod_database_last.csv', sep=';')

# Загрузка моделей и токенизаторов
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")

tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
model_roberta = AutoModel.from_pretrained("roberta-base")

KEYWORDS = {'Budve': 3, 'Bare': 3, 'Herceg Novi': 3, 'Podgorica':3,'apartment':2, 'plot of land':2, 'buy':2, 'to rent':2}  # пример ключевых слов и их весов

# Обработчик команды /start
async def start(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    username = user.username
    greeting = f'Привет, @{username}!' if username else 'Привет!'
    await update.message.reply_text(
        f'{greeting} Я ваш помощник-бот по поиску недвижимости в Черногории. '
        'Введите сообщение о ваших предпочтениях и мы подберем для вас вариант.'
    )


# Обработчик текстовых сообщений
async def handle_message(update: Update, context: CallbackContext) -> None:

    try:
        logger.info("Сообщение достигло обработчика.")
        user_message = update.message.text
        user_id = update.effective_user.id

        
        logger.info(f"Получено сообщение от пользователя {user_id}: {user_message}")

        translated_text = translate(user_message)
        logger.info(f"Перевод сообщения {translated_text}.")

        preprocessed__user_query= preprocess_user_query(translated_text)
        logger.info(f"Заменили слова {preprocessed__user_query}.")

        # Инициализация лемматизатора
        lemmatizer = WordNetLemmatizer()

        #preprocessed_text = preprocess_text(preprocessed__user_query)
        lemmatized_text = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(preprocessed__user_query)]
        # Преобразование списка обратно в строку
        lemmatized_text = ' '.join(lemmatized_text)
        logger.info(f"Лемматизация {len(lemmatized_text)}.")

          # Усиление текста ключевыми словами
        enhanced_text = enhance_text_with_keywords(lemmatized_text, KEYWORDS)
    
        # Усреднение эмбеддингов
        english_embedding= (get_embedding(enhanced_text,model_bert, tokenizer_bert) + get_embedding(enhanced_text, model_roberta, tokenizer_roberta))/2
            
        similarities = []
        for index, row in df_estate.iterrows():

            estate_embedding = row['embedding_ensemble']

            estate_embedding = estate_embedding.replace('[', '').replace(']', '').replace('\n', ' ')
            estate_embedding_array = np.fromstring(estate_embedding, sep=' ')
            # Преобразование NumPy массивов в тензоры PyTorch
            estate_embedding_tensor = torch.tensor(estate_embedding_array).type(torch.float32)
            english_embedding_tensor = torch.tensor(english_embedding).type(torch.float32)
            
            if english_embedding_tensor.dtype != estate_embedding_tensor.dtype:
                estate_embedding_tensor = estate_embedding.to(dtype=english_embedding_tensor.dtype)

            similarity = util.pytorch_cos_sim(english_embedding_tensor, estate_embedding_tensor)
            similarities.append((similarity, index))
            top_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
            

        for similarity, index in top_similarities:
            current_estate = df_estate.iloc[index]
            description = current_estate['Описание']
            link = current_estate['Ссылка']
            photo_urls = current_estate['Фотографии'].split('\n') if isinstance(current_estate['Фотографии'], str) else []

            photos = []
            for photo_url in photo_urls:
                if is_valid_image_url(photo_url):
                    try:
                        response = requests.get(photo_url)
                        if response.status_code == 200:
                            image_data = BytesIO(response.content)
                            photos.append(InputMediaPhoto(media=image_data))
                        if len(photos) == 4:
                            break
                    except Exception as e:
                        logger.error(f"Ошибка при загрузке изображения с URL {photo_url}: {e}")

            if photos:
                await context.bot.send_media_group(chat_id=update.effective_chat.id, media=photos)
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"{description}\nСсылка: {link}")
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Извините, не удалось загрузить фотографии.")

    except Exception as e:
    # В случае возникновения исключения, отправляем сообщение пользователю
        logger.error(f"Ошибка при обработке сообщения от пользователя {user_id}: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, 
                                        text="Произошла ошибка при обработке вашего сообщения. Сформулируйте ваш запрос по другому.")
