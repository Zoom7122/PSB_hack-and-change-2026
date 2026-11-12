"""Числовые факты: суммы, сроки, проценты, даты, номера пунктов и статей.

Зачем
=====
Самая опасная галлюцинация юридического ассистента — правдоподобное число:
«срок оплаты 90 дней», «порог 10 млн», «п. 4.7». Такие ошибки дёшево ловятся
без LLM: каждое число из утверждения должно встречаться в тексте фрагментов,
на которые утверждение ссылается.

Нормализация делает сравнение честным:
  «450 000» = «450000»; «12 млн» = «12 000 000»; «шести месяцев» = «6 месяцев»;
  «01.08.2024» = «1 августа 2024 г.»; «0,1%» = «0.1».

Не проверяются жёстко номера частей/абзацев/подпунктов («ч. 3 ст. 70»): в
выгрузке законов части не пронумерованы, и проверка давала бы ложные срабатывания.
Номера статей и пунктов с точкой («ст. 70», «п. 3.1») проверяются.
"""

from __future__ import annotations

import re
from datetime import date

MONTHS_GENITIVE = {
    "января": 1, "февраля": 2, "марта": 3, "апреля": 4, "мая": 5, "июня": 6,
    "июля": 7, "августа": 8, "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
}
_MONTHS_RE = "|".join(MONTHS_GENITIVE)

RE_DATE_NUMERIC = re.compile(r"(?<!\d)(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d)")
RE_DATE_WORDS = re.compile(rf"(?<!\d)(\d{{1,2}})\s+({_MONTHS_RE})\s+(\d{{4}})", re.IGNORECASE)

# Число с разрядами «8 400 000» / «0,1» / «3.1» / «6.2.1», плюс множитель.
_SPACE = "[   ]"
RE_NUMBER = re.compile(
    rf"(?<![\w])(\d{{1,3}}(?:{_SPACE}\d{{3}})+(?:[.,]\d+)?|\d+(?:[.,]\d+)*)"
    r"(?:\s*(млрд|миллиард\w*|млн|миллион\w*|тыс\.?|тысяч\w*))?",
    re.IGNORECASE,
)
_MULTIPLIERS = {"млрд": 10**9, "миллиард": 10**9, "млн": 10**6, "миллион": 10**6,
                "тыс": 10**3, "тысяч": 10**3}

# Числа после этих слов — ссылки на структуру документа, а не факты.
RE_SOFT_PREFIX = re.compile(
    r"(?:\b(?:ч|чч|абз|пп|подп|п|пункт\w*|част\w*|абзац\w*|подпункт\w*|№|N)\.?\s*)$",
    re.IGNORECASE,
)

RE_STEP_MARKER = re.compile(r"(?m)^\s*\d{1,2}[).]\s+(?!\d)")

_UNITS = {
    "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одному": 1, "одним": 1,
    "два": 2, "две": 2, "двух": 2, "двум": 2, "двумя": 2,
    "три": 3, "трех": 3, "трёх": 3, "трем": 3, "трём": 3, "тремя": 3,
    "четыре": 4, "четырех": 4, "четырёх": 4, "четырем": 4, "четырьмя": 4,
    "пять": 5, "пяти": 5, "пятью": 5, "шесть": 6, "шести": 6, "шестью": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9,
}
_TEENS = {
    "десять": 10, "десяти": 10, "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12, "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14, "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16, "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18, "девятнадцать": 19, "девятнадцати": 19,
}
_TENS = {
    "двадцать": 20, "двадцати": 20, "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40, "пятьдесят": 50, "пятидесяти": 50,
    "шестьдесят": 60, "шестидесяти": 60, "семьдесят": 70, "семидесяти": 70,
    "восемьдесят": 80, "восьмидесяти": 80, "девяносто": 90, "девяноста": 90,
    "сто": 100, "ста": 100,
}
RE_WORD = re.compile(r"[а-яё]+", re.IGNORECASE)


def _normalize_number(raw: str, multiplier: str | None) -> str:
    compact = re.sub(_SPACE, "", raw).replace(",", ".")
    if compact.count(".") > 1:          # «6.2.1» — номер пункта, не число
        return compact
    value = float(compact)
    if multiplier:
        key = next(k for k in _MULTIPLIERS if multiplier.lower().startswith(k))
        value *= _MULTIPLIERS[key]
    elif "." in compact:                # «3.1», «0.1» — сохраняем запись как есть
        return compact
    if value.is_integer():
        return str(int(value))
    return repr(value)


def _word_numbers(text: str) -> set[str]:
    """«сорока пяти» → 45, «шести» → 6, «тридцати» → 30."""
    found: set[str] = set()
    words = [w.lower() for w in RE_WORD.findall(text)]
    i = 0
    while i < len(words):
        word = words[i]
        if word in _TENS:
            value = _TENS[word]
            if i + 1 < len(words) and words[i + 1] in _UNITS and value < 100:
                value += _UNITS[words[i + 1]]
                i += 1
            found.add(str(value))
        elif word in _TEENS:
            found.add(str(_TEENS[word]))
        elif word in _UNITS:
            found.add(str(_UNITS[word]))
        i += 1
    return found


def parse_dates(text: str) -> list[date]:
    """Все даты из текста (цифрами и словами), в порядке появления."""
    found: list[tuple[int, date]] = []
    for match in RE_DATE_NUMERIC.finditer(text):
        day, month, year = (int(g) for g in match.groups())
        try:
            found.append((match.start(), date(year, month, day)))
        except ValueError:
            continue
    for match in RE_DATE_WORDS.finditer(text):
        day, month_name, year = match.groups()
        try:
            found.append((match.start(), date(int(year), MONTHS_GENITIVE[month_name.lower()], int(day))))
        except ValueError:
            continue
    return [value for _, value in sorted(found, key=lambda item: item[0])]


def _strip_dates(text: str) -> str:
    text = RE_DATE_NUMERIC.sub(" ", text)
    return RE_DATE_WORDS.sub(" ", text)


def numeric_facts(text: str, *, for_haystack: bool = False) -> set[str]:
    """Нормализованные числовые факты текста.

    for_haystack=False (утверждение): структурные ссылки («ч. 3», «п. 2»)
    пропускаются — их не проверяем жёстко.
    for_haystack=True (источник): берём всё, включая компоненты дат, чтобы
    «2024 году» в утверждении нашло «1 августа 2024 г.» в источнике.
    """
    facts: set[str] = set()
    if not for_haystack:
        # «1) Направить заявку…» — номер шага, а не факт.
        text = RE_STEP_MARKER.sub(" ", text)
    for value in parse_dates(text):
        facts.add(f"D:{value.isoformat()}")
        if for_haystack:
            facts.update({str(value.day), str(value.month), str(value.year)})
    rest = _strip_dates(text)
    for match in RE_NUMBER.finditer(rest):
        raw, multiplier = match.group(1), match.group(2)
        if not for_haystack and "." not in raw and "," not in raw:
            prefix = rest[max(0, match.start() - 12):match.start()]
            if RE_SOFT_PREFIX.search(prefix):
                continue
        try:
            facts.add(_normalize_number(raw, multiplier))
        except (ValueError, StopIteration):
            continue
        if for_haystack and multiplier is None and re.fullmatch(r"\d+", raw):
            facts.add(str(int(raw)))
    facts.update(_word_numbers(rest))
    return facts


def missing_facts(claim: str, haystack: str) -> list[str]:
    """Числовые факты утверждения, которых нет в тексте источников."""
    have = numeric_facts(haystack, for_haystack=True)
    missing = []
    for fact in sorted(numeric_facts(claim)):
        if fact in have:
            continue
        if fact.startswith("D:"):
            missing.append(date.fromisoformat(fact[2:]).strftime("%d.%m.%Y"))
        else:
            missing.append(fact)
    return missing


def amount_like(text: str) -> set[str]:
    """Факты, на которых строятся ловушки в вопросах: суммы, сроки, проценты, даты.

    Годы (1990–2100) сами по себе не считаем фактом-посылкой: «договор 2024 года»
    почти всегда совпадает с номером/датой документа и шумит.
    """
    result = set()
    for fact in numeric_facts(text):
        if fact.isdigit() and 1990 <= int(fact) <= 2100:
            continue
        result.add(fact)
    return result
