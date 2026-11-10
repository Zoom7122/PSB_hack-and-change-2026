"""Подготовка вопроса к поиску.

Зачем этот модуль нужен уже на шаге 3
=====================================
Половина демонстрационных вопросов — это короткий диалог:

    Пользователь: Мы арендуем склад у «Северного Склада». Планируем съезжать.
    Ассистент: Подскажите номер договора, чтобы я мог проверить условия возврата.
    Пользователь: А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?

Если искать по всему тексту целиком, поиск «размывается»: слова из первой
реплики и из реплики ассистента (в том числе служебные «пользователь»,
«ассистент», «подскажите») начинают конкурировать с главным — последним
вопросом. Проверено на практике: нужный пункт 6.3 оказывался на 12-м месте.

Правило из технического регламента прямо говорит, как надо:
«Если вопрос включает историю диалога, отвечайте на последнюю реплику
пользователя с учётом этой истории».

Поэтому: **ищем по последней реплике пользователя**, но **переносим в неё
сущности из истории** — номер договора, суммы, даты. В примере выше «А-08/2024»
стоит в последней реплике, а бывает наоборот: договор назвали в первой реплике,
а уточняющий вопрос пришёл без номера.

Это упрощённая версия шага 5: здесь работают только регулярные выражения,
без вызова модели. На шаге 5 добавится классификация типа вопроса и определение
ложных посылок.
"""

from __future__ import annotations

import re

# Маркеры ролей в диалоге. Платформа может прислать разные варианты написания.
RE_ROLE = re.compile(r"^\s*(Пользователь|Пользователь:|Ассистент|Ассистент:|User|Assistant|Клиент|Бот)\s*:",
                     re.IGNORECASE | re.MULTILINE)

# Сущности, которые надо переносить из истории в поисковый запрос.
RE_DOC_NUMBER = re.compile(r"(?<![А-Яа-яA-Za-z0-9])([А-ЯЁ]{1,3}[-–]?\s?\d{1,4}\s?/\s?\d{2,4})(?![\d/])")
RE_MONEY = re.compile(r"\d[\d\s]{1,12}(?:млн|тыс|руб|рублей|₽)")
RE_DATE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b")
RE_YEAR = re.compile(r"\b(20\d{2})\s*(?:год|г\.)?")


def is_dialog(question: str) -> bool:
    """Есть ли в вопросе история диалога."""
    return len(RE_ROLE.findall(question)) >= 2


def split_turns(question: str) -> list[tuple[str, str]]:
    """Разбить текст на реплики: [(роль, текст), …].

    Возвращает пустой список, если маркеров ролей нет.
    """
    matches = list(RE_ROLE.finditer(question))
    if not matches:
        return []

    turns: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        role = match.group(1).lower().rstrip(":")
        text = question[start:end].strip()
        if text:
            turns.append((role, text))
    return turns


def _is_user(role: str) -> bool:
    return role.startswith(("пользовател", "user", "клиент"))


def extract_entities(text: str) -> list[str]:
    """Вытащить сущности, важные для поиска: номера договоров, суммы, даты, годы."""
    found: list[str] = []
    for pattern in (RE_DOC_NUMBER, RE_DATE, RE_MONEY):
        found.extend(m.group(0).strip() for m in pattern.finditer(text))
    # Годы добавляем отдельно и только если перед нами не номер договора
    for match in RE_YEAR.finditer(text):
        found.append(match.group(1))
    # Убираем дубли, сохраняя порядок
    seen: set[str] = set()
    unique: list[str] = []
    for item in found:
        key = item.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def prepare_search_query(question: str) -> str:
    """Поисковый запрос из вопроса: последняя реплика пользователя + сущности из истории.

    Для недиалоговых вопросов возвращает текст без изменений.
    """
    turns = split_turns(question)
    if not turns:
        return question.strip()

    # Находим последнюю реплику пользователя
    last_user_index = None
    for i in range(len(turns) - 1, -1, -1):
        if _is_user(turns[i][0]):
            last_user_index = i
            break
    if last_user_index is None:
        # Диалог без реплик пользователя (маловероятно) — берём всё
        return question.strip()

    last_text = turns[last_user_index][1]
    history_text = " ".join(text for role, text in turns[:last_user_index] if not _is_user(role) or True)

    # Переносим сущности из истории, если их нет в последней реплике
    entities = extract_entities(history_text)
    missing = [e for e in entities if e.lower().replace(" ", "") not in last_text.lower().replace(" ", "")]
    if missing:
        return f"{last_text} {' '.join(missing)}"
    return last_text


def prepare_search_tokens(question: str) -> list[str]:
    """Токены для BM25 с весом: последняя реплика пользователя — дважды, история — один раз.

    Почему не только последняя реплика: в диалоге суть часто распределена.
    Пример из демо: «Подписали новый договор, оригинал пока у меня» →
    «В юротдел нужно отнести сразу или есть срок?». В последней реплике нет
    слов «оригинал» и «подписан», а без них нужный пункт регламента (передача
    оригинала в юротдел) не находится. Поэтому историю тоже учитываем, но с
    меньшим весом: BM25 считает каждый повтор токена отдельно, поэтому дубль
    последней реплики = двойной вес.
    """
    import textproc  # локальный импорт, чтобы модуль не тянул зависимость при импорте

    turns = split_turns(question)
    if not turns:
        return textproc.tokenize_query(question)

    last_user_text = ""
    for role, text in reversed(turns):
        if _is_user(role):
            last_user_text = text
            break

    history_text = " ".join(text for _, text in turns)
    tokens = textproc.tokenize_query(last_user_text) * 2 + textproc.tokenize_query(history_text)
    return tokens


def describe(question: str) -> str:
    """Отладочный вывод: как вопрос превратился в поисковый запрос."""
    turns = split_turns(question)
    lines = [f"диалог: {'да' if turns else 'нет'}"]
    if turns:
        for i, (role, text) in enumerate(turns, 1):
            lines.append(f"  {i}. [{role}] {text[:80]}")
    prepared = prepare_search_query(question)
    lines.append(f"поисковый запрос: {prepared[:120]}")
    entities = extract_entities(question)
    if entities:
        lines.append(f"сущности: {entities}")
    return "\n".join(lines)


if __name__ == "__main__":
    samples = [
        "Какая арендная плата по договору с ООО «Северный Склад»?",
        "Пользователь: Мы арендуем склад у «Северного Склада». Планируем съезжать.\n"
        "Ассистент: Подскажите номер договора, чтобы я мог проверить условия возврата.\n"
        "Пользователь: А-08/2024. Можно просто сдать ключи или обязательно оформлять акт?",
        "Пользователь: Подскажите гарантию по договору с «Альфой».\n"
        "Ассистент: Уточните, пожалуйста, поставщик — «Альфа-Трейд» или «Альфа-Трейдинг»?\n"
        "Пользователь: «Альфа-Трейдинг», у них компьютеры покупали. На сколько гарантия?",
    ]
    for sample in samples:
        print("-" * 80)
        print(describe(sample))
