import torch
import numpy as np
import nltk
import spacy
from nltk.corpus import wordnet
from transformers import AutoTokenizer, AutoModel, MarianMTModel, MarianTokenizer

# Функция для перевода текста
model_name = 'Helsinki-NLP/opus-mt-ru-en'
tokenizer_translate = MarianTokenizer.from_pretrained(model_name)
model_translate = MarianMTModel.from_pretrained(model_name)

# Загрузка моделей и токенизаторов
tokenizer_bert = AutoTokenizer.from_pretrained("bert-base-uncased")
model_bert = AutoModel.from_pretrained("bert-base-uncased")

# Загрузка модели NER
nlp = spacy.load("en_core_web_sm")  # или другая модель NER

def translate(text):
    inputs = tokenizer_translate(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        translated = model_translate.generate(**inputs)
    return tokenizer_translate.decode(translated[0], skip_special_tokens=True)


# Функция для получения эмбеддингов от модели
def get_embedding(text, model, tokenizer):
    
    if not text:
        # Возвращаем нулевой вектор или другое подходящее значение
        return np.zeros(shape=(768,))
    
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model_bert(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# Функция для определения части речи слова
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

