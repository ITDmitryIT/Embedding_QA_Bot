# Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings('ignore')
import mwclient  # Для работы с MediaWiki API
import mwparserfromhell  # Для парсинга вики-текста
import pandas as pd  # Для работы с DataFrame
import re  # Для регулярных выражений
import tiktoken  # Для подсчета токенов
from openai import OpenAI  # Для вычисления эмбедингов
import numpy as np  # Для работы с массивами
from sklearn.metrics.pairwise import cosine_similarity  # Для косинусного сходства
import os
import getpass  # Для безопасного ввода API ключа

# Задаем категорию и англоязычную версию Википедии для поиска
CATEGORY_TITLE = "Category:History of artificial intelligence"
WIKI_SITE = "en.wikipedia.org"

# Собираем заголовки всех статей
def titles_from_category(
    category: mwclient.listing.Category,  # Категория статей
    max_depth: int  # Глубина вложенности
) -> set[str]:
    """Возвращает набор заголовков страниц в данной категории Википедии и ее подкатегориях."""
    titles = set()  # Множество для хранения заголовков
    for cm in category.members():  # Перебираем объекты категории
        if type(cm) == mwclient.page.Page:  # Если объект - страница
            titles.add(cm.name)  # Добавляем имя страницы
        elif isinstance(cm, mwclient.listing.Category) and max_depth > 0:  # Если объект - подкатегория
            deeper_titles = titles_from_category(cm, max_depth=max_depth - 1)  # Рекурсивно обрабатываем подкатегорию
            titles.update(deeper_titles)  # Добавляем заголовки из подкатегории
    return titles

# Инициализация объекта MediaWiki
site = mwclient.Site(WIKI_SITE)
category_page = site.pages[CATEGORY_TITLE]

# Получение множества всех заголовков категории с вложенностью на один уровень
titles = titles_from_category(category_page, max_depth=1)

print(f"Создано {len(titles)} заголовков статей в категории {CATEGORY_TITLE}.")

# Задаем секции, которые будут отброшены при парсинге статей
SECTIONS_TO_IGNORE = [
    "See also",
    "References",
    "External links",
    "Further reading",
    "Footnotes",
    "Bibliography",
    "Sources",
    "Citations",
    "Literature",
    "Footnotes",
    "Notes and references",
    "Photo gallery",
    "Works cited",
    "Photos",
    "Gallery",
    "Notes",
    "References and sources",
    "References and notes",
]

# Функция возвращает список всех вложенных секций для заданной секции страницы Википедии
def all_subsections_from_section(
    section: mwparserfromhell.wikicode.Wikicode,  # Текущая секция
    parent_titles: list[str],  # Заголовки родителя
    sections_to_ignore: set[str],  # Секции, которые необходимо проигнорировать
) -> list[tuple[list[str], str]]:
    """
    Из раздела Википедии возвращает список всех вложенных секций.
    Каждый подраздел представляет собой кортеж, где:
      - первый элемент представляет собой список родительских секций, начиная с заголовка страницы
      - второй элемент представляет собой текст секции
    """
    headings = [str(h) for h in section.filter_headings()]
    title = headings[0]
    if title.strip("=" + " ") in sections_to_ignore:
        return []
    titles = parent_titles + [title]
    full_text = str(section)
    section_text = full_text.split(title)[1]
    if len(headings) == 1:
        return [(titles, section_text)]
    else:
        first_subtitle = headings[1]
        section_text = section_text.split(first_subtitle)[0]
        results = [(titles, section_text)]
        for subsection in section.get_sections(levels=[len(titles) + 1]):
            results.extend(all_subsections_from_section(subsection, titles, sections_to_ignore))
        return results

# Функция возвращает список всех секций страницы, за исключением тех, которые отбрасываем
def all_subsections_from_title(
    title: str,  # Заголовок статьи Википедии, которую парсим
    sections_to_ignore: set[str] = SECTIONS_TO_IGNORE,  # Секции, которые игнорируем
    site_name: str = WIKI_SITE,  # Ссылка на сайт википедии
) -> list[tuple[list[str], str]]:
    site = mwclient.Site(site_name)
    page = site.pages[title]
    text = page.text()
    parsed_text = mwparserfromhell.parse(text)
    headings = [str(h) for h in parsed_text.filter_headings()]
    if headings:
        summary_text = str(parsed_text).split(headings[0])[0]
    else:
        summary_text = str(parsed_text)
    results = [([title], summary_text)]
    for subsection in parsed_text.get_sections(levels=[2]):
        results.extend(all_subsections_from_section(subsection, [title], sections_to_ignore))
    return results

# Разбивка статей на секции
wikipedia_sections = []
for title in titles:
    wikipedia_sections.extend(all_subsections_from_title(title))

print(f"Найдено {len(wikipedia_sections)} секций на {len(titles)} страницах.")

# Функция для удаления заголовков секций
def remove_section_titles(text):
    # Удаляем заголовки в формате ==Title==, ===Title=== и т.д.
    text = re.sub(r"={2,}.*?={2,}", "", text)
    # Удаляем лишние пробелы и пустые строки
    text = re.sub(r"\n\s*\n", "\n", text).strip()
    return text

# Функция для очистки текста от вики-разметки
def clean_wikipedia_text(text):
    parsed_text = mwparserfromhell.parse(text)
    # Удаляем все ссылки, шаблоны и другие элементы разметки
    plain_text = parsed_text.strip_code()
    return plain_text

# Очистка текста секции от ссылок <ref>xyz</ref>, начальных и конечных пробелов
def clean_section(section: tuple[list[str], str]) -> tuple[list[str], str]:
    titles, text = section
    text = re.sub(r"<ref.*?</ref>", "", text)  # Удаляем ссылки
    text = remove_section_titles(text)  # Удаляем заголовки секций
    text = clean_wikipedia_text(text)  # Очищаем от вики-разметки
    text = text.strip()  # Удаляем лишние пробелы
    return (titles, text)

# Применим функцию очистки ко всем секциям с помощью генератора списков
wikipedia_sections = [clean_section(ws) for ws in wikipedia_sections]

# Отфильтруем короткие и пустые секции
def keep_section(section: tuple[list[str], str]) -> bool:
    titles, text = section
    if len(text) < 16:
        return False
    else:
        return True

original_num_sections = len(wikipedia_sections)
wikipedia_sections = [ws for ws in wikipedia_sections if keep_section(ws)]

print(f"Отфильтровано {original_num_sections - len(wikipedia_sections)} секций, осталось {len(wikipedia_sections)} секций.")

# Функция подсчета токенов
def num_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Функция разделения строк
def halved_by_delimiter(string: str, delimiter: str = "\n") -> list[str, str]:
    chunks = string.split(delimiter)
    if len(chunks) == 1:
        return [string, ""]
    elif len(chunks) == 2:
        return chunks
    else:
        total_tokens = num_tokens(string)
        halfway = total_tokens // 2
        best_diff = halfway
        for i, chunk in enumerate(chunks):
            left = delimiter.join(chunks[: i + 1])
            left_tokens = num_tokens(left)
            diff = abs(halfway - left_tokens)
            if diff >= best_diff:
                break
            else:
                best_diff = diff
        left = delimiter.join(chunks[:i])
        right = delimiter.join(chunks[i:])
        return [left, right]

# Функция обрезает строку до максимально разрешенного числа токенов
def truncated_string(
    string: str,
    model: str,
    max_tokens: int,
    print_warning: bool = True,
) -> str:
    encoding = tiktoken.encoding_for_model(model)
    encoded_string = encoding.encode(string)
    truncated_string = encoding.decode(encoded_string[:max_tokens])
    if print_warning and len(encoded_string) > max_tokens:
        print(f"Предупреждение: Строка обрезана с {len(encoded_string)} токенов до {max_tokens} токенов.")
    return truncated_string

# Функция делит секции статьи на части по максимальному числу токенов
def split_strings_from_subsection(
    subsection: tuple[list[str], str],
    max_tokens: int = 1000,
    model: str = "gpt-3.5-turbo",
    max_recursion: int = 5,
) -> list[str]:
    titles, text = subsection
    string = "\n\n".join(titles + [text])
    num_tokens_in_string = num_tokens(string)
    if num_tokens_in_string <= max_tokens:
        return [string]
    elif max_recursion == 0:
        return [truncated_string(string, model=model, max_tokens=max_tokens)]
    else:
        titles, text = subsection
        for delimiter in ["\n\n", "\n", ". "]:
            left, right = halved_by_delimiter(text, delimiter=delimiter)
            if left == "" or right == "":
                continue
            else:
                results = []
                for half in [left, right]:
                    half_subsection = (titles, half)
                    half_strings = split_strings_from_subsection(
                        half_subsection,
                        max_tokens=max_tokens,
                        model=model,
                        max_recursion=max_recursion - 1,
                    )
                    results.extend(half_strings)
                return results
    return [truncated_string(string, model=model, max_tokens=max_tokens)]

# Делим секции на части
MAX_TOKENS = 1600
wikipedia_strings = []
for section in wikipedia_sections:
    titles, text = section
    cleaned_text = clean_wikipedia_text(text)  # Очищаем текст
    wikipedia_strings.extend(split_strings_from_subsection((titles, cleaned_text), max_tokens=MAX_TOKENS))

print(f"{len(wikipedia_sections)} секций Википедии поделены на {len(wikipedia_strings)} строк.")

# Токенизация и сохранение результата
os.environ["OPENAI_API_KEY"] = getpass.getpass("Введите OpenAI API Key:")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Функция отправки chatGPT строки для ее токенизации (вычисления эмбедингов)
def get_embedding(text, model="text-embedding-ada-002"):
    return openai_client.embeddings.create(input=[text], model=model).data[0].embedding

# Создаем DataFrame с текстом и эмбедингами
df = pd.DataFrame({"text": wikipedia_strings})
df['embedding'] = df.text.apply(lambda x: get_embedding(x, model='text-embedding-ada-002'))

# Сохраняем результат в CSV файл
SAVE_PATH = "./history_of_ai.csv"
df.to_csv(SAVE_PATH, index=False)

print(f"База знаний сохранена в файл {SAVE_PATH}.")
