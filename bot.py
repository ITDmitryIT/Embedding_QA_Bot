# Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings('ignore')
import pandas as pd  # Для работы с DataFrame
from sklearn.metrics.pairwise import cosine_similarity  # Для косинусного сходства
from openai import OpenAI  # Для вычисления эмбедингов
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.filters.command import Command
import os
import getpass
import asyncio
import nest_asyncio
from deep_translator import GoogleTranslator  # Для перевода текста

# Активируем nest_asyncio для работы с уже запущенным циклом событий
nest_asyncio.apply()

# Загрузка базы знаний из CSV файла
SAVE_PATH = "./history_of_ai.csv"
df = pd.read_csv(SAVE_PATH)

# Преобразование строки embedding обратно в массив numpy
df['embedding'] = df['embedding'].apply(lambda x: eval(x))

# Функция для получения эмбединга текста
def get_embedding(text, model="text-embedding-ada-002"):
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

# Функция для поиска наиболее подходящего ответа
def find_best_answer(question, df, threshold=0.75):
    question_embedding = get_embedding(question)
    # Вычисляем косинусное сходство между вопросом и всеми записями в базе знаний
    similarities = df['embedding'].apply(lambda x: cosine_similarity([question_embedding], [x])[0][0])

    # Находим индекс наиболее подходящего ответа
    best_index = similarities.idxmax()
    best_similarity = similarities[best_index]

    # Если сходство ниже порогового значения, возвращаем сообщение об отсутствии ответа
    if best_similarity < threshold:
        return "Извините, я не могу найти точный ответ на ваш вопрос."

    # Возвращаем текст наиболее подходящей записи
    return df.iloc[best_index]['text']

# Функция для перевода текста на русский
def translate_to_russian(text):
    translator = GoogleTranslator(source='en', target='ru')
    translated_text = translator.translate(text)  # Синхронный вызов
    return translated_text  # Возвращаем переведённый текст

# Инициализация Telegram-бота
os.environ["OPENAI_API_KEY"] = getpass.getpass("Введите OpenAI API Key:")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
bot_token = getpass.getpass("Введите токен вашего Telegram-бота:")
bot = Bot(token=bot_token)
dp = Dispatcher()

# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer("Привет! Я бот, который может отвечать на вопросы об истории искусственного интеллекта. "
                         "Напишите ваш вопрос или используйте команду /help для получения информации.")

# Обработчик команды /help
@dp.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "Информация о базе знаний:\n"
        "Тематика: История искусственного интеллекта.\n"
        f"Число записей в базе знаний: {len(df)}\n"
        "Пример запроса: 'Кто создал первый нейронный компьютер?'"
    )
    await message.answer(help_text)

# Обработчик текстовых сообщений
@dp.message()
async def handle_message(message: Message):
    question = message.text
    answer = find_best_answer(question, df)  # Ищем ответ в базе знаний

    # Если ответ слишком длинный, используем OpenAI для его сокращения
    if len(answer.split()) > 50:  # Ограничение длины ответа (например, 50 слов)
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Сократи текст до нескольких предложений, сохраняя ключевую информацию."},
                {"role": "user", "content": answer}
            ]
        )
        answer = response.choices[0].message.content

    # Переводим ответ на русский язык
    translated_answer = translate_to_russian(answer)  # Вызываем синхронную функцию

    # Отправляем переведённый ответ пользователю
    await message.answer(translated_answer)

# Запуск бота
async def main():
    await dp.start_polling(bot)

# Запуск асинхронного цикла в Google Colab
if __name__ == "__main__":
    asyncio.run(main())
