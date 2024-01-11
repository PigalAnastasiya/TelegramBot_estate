
import pandas as pd
import torch
import numpy as np
import spacy
from bot_handlers import start, handle_message
from nltk.stem import PorterStemmer
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from my_own_secrets import TOKEN_ID

# Основная функция для запуска бота
def main() -> None:
    TOKEN = TOKEN_ID
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.run_polling()

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
