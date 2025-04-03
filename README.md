## Embedding_QA_Bot
Telegram-бот, который помогает пользователям находить ответы на вопросы об истории искусственного интеллекта. Бот использует современные технологии, такие как эмбеддинги текста (text embeddings), косинусное сходство и машинный перевод, чтобы предоставлять точные и понятные ответы
## Основные возможности:
Поиск ответов в базе знаний: Бот работает с предварительно созданной базой знаний, содержащей информацию об истории ИИ.
Автоматический перевод: Все ответы автоматически переводятся на русский язык для удобства пользователей.
Сокращение длинных ответов: Если ответ слишком объемный, бот сокращает его, сохраняя ключевую информацию, с помощью модели GPT.
Простота использования: Достаточно задать вопрос в чате, и бот предоставит релевантный ответ.
## Технологии:
OpenAI API: Для вычисления эмбеддингов и сокращения текста.
Cosine Similarity: Для поиска наиболее подходящих ответов в базе знаний.
Deep Translator: Для перевода текста на русский язык.
Aiogram Framework: Для создания Telegram-бота.
Pandas: Для работы с базой знаний в формате CSV.
## Как использовать:
Запустите бота командой 
```bash
 /start. 
Задайте вопрос о истории искусственного интеллекта.
Получите ответ на русском языке!
## Цель проекта:
Проект создан для образовательных целей и демонстрирует, как можно использовать современные технологии ИИ для создания полезных инструментов, таких как чат-боты с базой знаний.

## База знаний:
Тематика: История искусственного интеллекта.
Число записей: {len(df)} (автоматически подставляется из вашего DataFrame).
Пример запроса: "Кто создал первый нейронный компьютер?"
## Установка и запуск:
Склонируйте репозиторий.
Установите зависимости: pip install -r requirements.txt.
Настройте переменные окружения (OPENAI_API_KEY и токен Telegram-бота).
Запустите бота: python bot.py.
