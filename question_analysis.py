"""Шаг 5. Понимание вопроса: сущности, тип вопроса, выбор редакции.

Зачем это нужно
===============
Поиск на шаге 4 даёт hit@5 = 1.00: нужный фрагмент всегда в контексте. Значит
дальше ошибаться будет не поиск, а интерпретация вопроса. Половина
демо-вопросов — ловушки:

  * «Почему арендная плата 500 000?» — ложная посылка: в договоре 450 000.
    Если ответить по посылке, ассистент соврёт.
  * «Договор на 12 млн заключали в августе 2024, нужно ли одобрение?» — вопрос
    про регламент, и применима редакция, действовавшая на дату события.
  * «Альфа-Трейд» и «Альфа-Трейдинг» — два разных контрагента с разными ИНН.
  * «Какой размер пособия по беременности?» — в базе нет ответа, нужен отказ.

Что делает этот модуль
======================
1. Извлекает сущности регулярками: суммы, ИНН, номера договоров, даты, пункты,
   статьи законов, контрагентов из метаданных базы.
2. Определяет тип вопроса правилами (без модели — быстро и бесплатно), с
   возможностью уточнить одним дешёвым вызовом LLM.
3. Находит «заявленные факты» — числа и утверждения, поданные как данность,
   чтобы на шаге 6 сверить их с документами, а не поверить на слово.
4. Выбирает применимую редакцию документа по дате события из вопроса.

Чего этот модуль НЕ делает
==========================
Не ищет и не отвечает. Только разбирает вопрос. Это важно: разбор можно
протестировать отдельно — 20 офлайновых проверок в check_step5.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import config
import query_prep

# ---------------------------------------------------------------------------
# СЛОВАРИ И РЕГУЛЯРКИ
# ---------------------------------------------------------------------------

MONTHS = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

# «450 000», «12 млн», «1 200 000 рублей», «500000»
RE_AMOUNT = re.compile(
    r"\b(\d{1,3}(?:[ \u00a0]\d{3})+|\d+)\s*(млн|млрд|тыс|тысяч|миллион\w*|миллиард\w*)?\b",
    re.IGNORECASE,
)
RE_MULTIPLIER = {"млн": 1_000_000, "миллион": 1_000_000, "млрд": 1_000_000_000,
                 "миллиард": 1_000_000_000, "тыс": 1_000, "тысяч": 1_000}

RE_INN = re.compile(r"\bИНН\s*[:№]?\s*(\d{10}|\d{12})\b", re.IGNORECASE)
RE_INN_BARE = re.compile(r"\b(\d{10}|\d{12})\b")

# Номера договоров: А-08/2024, П-17/2024, ПТ-05/2023, К-12, ЮО-7
RE_CONTRACT_NUMBER = re.compile(r"\b([А-ЯЁA-Z]{1,4}-\d{1,4}/\d{4}|[А-ЯЁA-Z]{1,4}-\d{1,4})\b")

RE_DATE_FULL = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b")
# «в августе 2024», «август 2024 года», «за август 2024»
RE_DATE_MONTH_YEAR = re.compile(
    r"\b(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йяе]|июн\w*|июл\w*|август\w*|"
    r"сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)\s+(\d{4})",
    re.IGNORECASE,
)
RE_CLAUSE = re.compile(r"\b(?:п\.|пункт\w*|параграф\w*)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
RE_ARTICLE = re.compile(r"\b(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)

# Ключевые слова типов вопросов
REFUSAL_MARKERS = (
    "пособи", "декрет", "беременност", "алимент", "ипотек", "страховк",
    "налоговый вычет", "кредитн", "отпускны", "субсиди", "виз", "таможн",
)
MULTI_SOURCE_MARKERS = (
    "кто должен", "кто запускает", "согласован", "порядок согласования",
    "одобрени", "утверждени", "регламент",
)
DIALOG_SPLIT = re.compile(r"(?:^|\n)\s*(?:Пользователь|Ассистент|User|Assistant)\s*:", re.IGNORECASE)


# ---------------------------------------------------------------------------
# СТРУКТУРЫ
# ---------------------------------------------------------------------------

@dataclass
class Entities:
    """Сущности, извлечённые из вопроса."""
    amounts: list[int] = field(default_factory=list)
    inns: list[str] = field(default_factory=list)
    contract_numbers: list[str] = field(default_factory=list)
    dates: list[date] = field(default_factory=list)
    clauses: list[str] = field(default_factory=list)
    articles: list[str] = field(default_factory=list)
    counterparties: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((self.amounts, self.inns, self.contract_numbers, self.dates,
                        self.clauses, self.articles, self.counterparties))


@dataclass
class Claim:
    """Факт из вопроса, который надо сверить с документами.

    asserted=True — подан как утверждение в провокационном вопросе
    («почему плата 500 000?»). Такой факт почти наверняка и есть ловушка.
    asserted=False — обычное упоминание («договор на 12 млн»); сверить всё
    равно надо, но тип вопроса оно не меняет.
    """
    value: str
    kind: str          # amount | inn | contract_number | date
    asserted: bool = False

    def __str__(self) -> str:
        return f"{self.kind}: {self.value}" + (" (утверждение)" if self.asserted else "")


@dataclass
class QuestionAnalysis:
    """Результат разбора вопроса."""
    raw: str
    search_text: str = ""
    entities: Entities = field(default_factory=Entities)
    kind: str = "factoid"                 # factoid | false_premise | multi_source |
                                          # edition_dependent | dialog | out_of_scope
    is_dialog: bool = False
    claims: list[Claim] = field(default_factory=list)
    event_date: date | None = None
    preferred_doc_ids: list[str] = field(default_factory=list)
    superseded_doc_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        parts = [f"тип={self.kind}"]
        if self.is_dialog:
            parts.append("диалог")
        if self.entities.counterparties:
            parts.append("контрагенты: " + ", ".join(self.entities.counterparties))
        if self.entities.contract_numbers:
            parts.append("договоры: " + ", ".join(self.entities.contract_numbers))
        if self.entities.amounts:
            parts.append("суммы: " + ", ".join(f"{a:,}".replace(",", " ") for a in self.entities.amounts))
        if self.event_date:
            parts.append(f"дата события: {self.event_date:%d.%m.%Y}")
        if self.claims:
            parts.append("проверить: " + "; ".join(map(str, self.claims)))
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# ЗАГРУЗКА МЕТАДАННЫХ БАЗЫ
# ---------------------------------------------------------------------------

def load_doc_meta(path: Path | str | None = None) -> dict[str, dict]:
    """Метаданные документов: контрагенты, редакции, статусы, номера, ИНН.

    Источник 1 — artifacts/docs.json (его пишет build_chunks.py на шаге 2).
    Если файла нет (не выполняли шаг 2, удалили artifacts/, другой каталог),
    метаданные собираются заново из базы знаний прямо в памяти: это занимает
    около секунды и не требует сети.

    Почему важно не возвращать пустой словарь. Без метаданных разбор вопросов
    не падает, а тихо деградирует: не находятся контрагенты, не выбирается
    редакция. Проверки при этом показывают загадочные пустые списки, и
    приходится искать причину в коде вместо того, чтобы увидеть её сразу.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend([
            Path(config.ARTIFACTS_DIR) / "docs.json",
            Path(config.BASE_DIR) / "artifacts" / "docs.json",
            Path(config.BASE_DIR) / "build" / "docs.json",
        ])

    for candidate in candidates:
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"[question_analysis] {candidate} не читается ({exc}); собираю метаданные заново")
                break
            if data:
                return data

    # запасной путь: пересобрать метаданные из базы знаний
    try:
        import chunking
        import ingest

        archive = ingest.find_knowledge_base(config.DATA_DIR)
        documents = ingest.load_documents_from_zip(archive)
        _, metas = chunking.build_chunks(documents, verbose=False)
        rebuilt = {doc_id: meta.__dict__ for doc_id, meta in metas.items()}
        print(f"[question_analysis] artifacts/docs.json не найден — "
              f"метаданные собраны из базы знаний ({len(rebuilt)} документов). "
              f"Создать файл: python build_chunks.py")
        return rebuilt
    except Exception as exc:      # noqa: BLE001 — без метаданных шаг 5 не работает
        print(f"[question_analysis] не удалось получить метаданные документов: {exc}")
        return {}


# ---------------------------------------------------------------------------
# ИЗВЛЕЧЕНИЕ СУЩНОСТЕЙ
# ---------------------------------------------------------------------------

def parse_amounts(text: str) -> list[int]:
    """Найти суммы: «450 000», «12 млн», «1 200 000 рублей»."""
    found: list[int] = []
    for match in RE_AMOUNT.finditer(text):
        digits, suffix = match.group(1), (match.group(2) or "").lower()
        number = int(re.sub(r"[ \u00a0]", "", digits))
        if suffix:
            for key, factor in RE_MULTIPLIER.items():
                if suffix.startswith(key[:4]):
                    number *= factor
                    break
        # отсекаем то, что деньгами не является:
        #   * числа меньше 1000 — это пункты, дни, штуки;
        #   * годы (1900–2100), записанные без разбивки на разряды —
        #     в «августе 2024 года» 2024 попал бы в суммы
        grouped = " " in digits or "\u00a0" in digits
        if number >= 1000 and not (1900 <= number <= 2100 and not grouped):
            found.append(number)
    return sorted(set(found))


def parse_inns(text: str) -> list[str]:
    """ИНН из вопроса: с пометкой «ИНН» или просто 10/12 цифр подряд."""
    inns = set(RE_INN.findall(text))
    if not inns:
        # без пометки берём только 10/12-значные, но не те, что уже суммы
        for candidate in RE_INN_BARE.findall(text):
            if not any(str(amount) == candidate for amount in parse_amounts(text)):
                inns.add(candidate)
    return sorted(inns)


def parse_contract_numbers(text: str) -> list[str]:
    """Номера договоров: А-08/2024, П-17/2024, ПТ-05/2023, К-12, ЮО-7."""
    numbers = [m.group(1) for m in RE_CONTRACT_NUMBER.finditer(text)]
    # отбрасываем «152-ФЗ»-подобное: там номер закона без буквы-кода
    return sorted({n for n in numbers if re.search(r"\d", n)})


def _month_to_number(word: str) -> int | None:
    lowered = word.lower()
    for stem, number in MONTHS.items():
        if lowered.startswith(stem):
            return number
    return None


def parse_dates(text: str) -> list[date]:
    """Даты в любом из встречающихся видов: 01.08.2024 или «августе 2024»."""
    dates: list[date] = []
    for day, month, year in RE_DATE_FULL.findall(text):
        try:
            dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    for month_word, year in RE_DATE_MONTH_YEAR.findall(text):
        month = _month_to_number(month_word)
        if month:
            dates.append(date(int(year), month, 1))
    return sorted(set(dates))


def find_counterparties(text: str, meta: dict[str, dict]) -> list[str]:
    """Контрагенты из метаданных базы, упомянутые в вопросе.

    Здесь важна аккуратность: «Альфа-Трейд» — подстрока «Альфа-Трейдинг».
    Поэтому сначала находим самые長 длинные совпадения, а короткие отбрасываем,
    если они целиком входят в найденное длинное.
    """
    known = sorted({d.get("counterparty", "") for d in meta.values() if d.get("counterparty")},
                   key=len, reverse=True)
    lowered = text.lower().replace("ё", "е").replace("«", "").replace("»", "")
    hits: list[str] = []
    for name in known:
        needle = name.lower().replace("ё", "е")
        if needle and needle in lowered:
            hits.append(name)
    # отбрасываем тех, чьё имя — подстрока более длинного найденного
    result = []
    for name in hits:
        long_name = next((other for other in hits
                          if other != name and name.lower() in other.lower()), None)
        if long_name is None:
            result.append(name)
    return result


def detect_dialog(text: str) -> bool:
    """Несколько реплик в одном вопросе — значит контекст надо собрать из истории."""
    return len(DIALOG_SPLIT.findall(text)) >= 2 or text.count("\n") >= 2


# ---------------------------------------------------------------------------
# ЛОЖНЫЕ ПОСЫЛКИ
# ---------------------------------------------------------------------------

def find_claims(text: str, entities: Entities) -> list[Claim]:
    """Числа и реквизиты, поданные в вопросе как факт.

    Зачем: «Почему арендная плата 500 000?» — это утверждение, а не вопрос о
    сумме. Правильный ответ: «в договоре 450 000, вот пункт». Если модель не
    знает, что 500 000 — посылка, которую надо проверить, она подхватит её и
    объяснит несуществующее условие.

    Эвристика: вопрос с вопросительным словом «почему / разве / неужели» и
    числом или реквизитом почти всегда содержит проверяемое утверждение.
    """
    lowered = text.lower()
    is_provocative = any(word in lowered for word in
                         ("почему", "разве", "неужели", "правда ли", "ведь"))

    claims: list[Claim] = []
    claims.extend(Claim(f"{a:,}".replace(",", " "), "amount", is_provocative)
                  for a in entities.amounts)
    claims.extend(Claim(inn, "inn", is_provocative) for inn in entities.inns)
    claims.extend(Claim(number, "contract_number", is_provocative)
                  for number in entities.contract_numbers)
    # дата, поданная как обстоятельство («заключали в августе 2024»)
    if re.search(r"заключ|подпис|случил|произош", lowered):
        claims.extend(Claim(f"{found:%d.%m.%Y}", "date", is_provocative)
                      for found in entities.dates)
    return claims


# ---------------------------------------------------------------------------
# РЕДАКЦИИ ДОКУМЕНТОВ
# ---------------------------------------------------------------------------

def _doc_family(meta_row: dict) -> str:
    """Семейство документа: название без пометки о редакции и статусе."""
    title = meta_row.get("citation_title", "")
    title = re.sub(r"\(ред\..*?\)", "", title)
    title = re.sub(r"утратил\w* силу", "", title)
    return re.sub(r"[\s,]+", " ", title).strip().lower()


def choose_editions(entities: Entities, meta: dict[str, dict],
                    event_date: date | None) -> tuple[list[str], list[str], list[str]]:
    """Выбрать применимые редакции документов.

    Возвращает (приоритетные doc_id, устаревшие doc_id, пояснения).

    Логика — пометка, а не фильтр (как договорились): устаревшую редакцию мы не
    выбрасываем, а помечаем. Причина: если на дату события действовала именно
    старая редакция, ответ надо давать по ней, но полезно и показать, что
    сейчас правило другое.
    """
    notes: list[str] = []
    preferred: list[str] = []
    superseded: list[str] = []

    # группируем документы по семейству
    families: dict[str, list[tuple[str, dict]]] = {}
    for doc_id, row in meta.items():
        families.setdefault(_doc_family(row), []).append((doc_id, row))

    for family, members in families.items():
        if len(members) < 2:
            continue
        # редакция задана в поле edition (10.01.2023) или в заголовке
        dated: list[tuple[date, str, dict]] = []
        for doc_id, row in members:
            raw = row.get("edition") or ""
            parsed = parse_dates(raw)
            if parsed:
                dated.append((parsed[0], doc_id, row))
        if not dated:
            continue
        dated.sort()

        if event_date:
            applicable = [item for item in dated if item[0] <= event_date]
            chosen = applicable[-1] if applicable else dated[0]
            notes.append(
                f"на {event_date:%d.%m.%Y} действовала редакция {chosen[0]:%d.%m.%Y} "
                f"({chosen[2].get('citation_title', '')})")
        else:
            valid = [item for item in dated if item[2].get("status") == "действует"]
            chosen = valid[-1] if valid else dated[-1]

        preferred.append(chosen[1])
        for item in dated:
            if item[1] != chosen[1]:
                superseded.append(item[1])
                notes.append(f"редакция {item[0]:%d.%m.%Y} помечена как неприменимая к этому вопросу")

    return preferred, superseded, notes


# ---------------------------------------------------------------------------
# ТИП ВОПРОСА
# ---------------------------------------------------------------------------

def detect_kind(text: str, entities: Entities, meta: dict[str, dict]) -> str:
    """Тип вопроса по правилам. Без вызова модели: быстро и бесплатно.

    Порядок важен: проверяем от самого опасного к обычному.
    """
    lowered = text.lower().replace("ё", "е")

    # 1. базы не хватает — вопрос про то, чего в документах нет
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return "out_of_scope"

    # 2. дата события — нужна применимая редакция
    #    Проверяем ДО ложной посылки: «договор на 12 млн заключали в августе
    #    2024» — это не провокация про сумму, это вопрос про редакцию регламента.
    if entities.dates and re.search(r"заключ|подпис|случил|произош|в \w+ \d{4}", lowered):
        return "edition_dependent"

    # 3. ложная посылка — есть УТВЕРЖДЕНИЕ, поданное как данность
    if any(claim.asserted for claim in find_claims(text, entities)):
        return "false_premise"

    # 4. нужен не один документ: договор + регламент, или два контрагента
    if any(marker in lowered for marker in MULTI_SOURCE_MARKERS) or len(entities.counterparties) > 1:
        return "multi_source"

    # 5. диалог с историей
    if detect_dialog(text):
        return "dialog"

    return "factoid"


# ---------------------------------------------------------------------------
# ГЛАВНАЯ ФУНКЦИЯ
# ---------------------------------------------------------------------------

def analyze(question: str, meta: dict[str, dict] | None = None) -> QuestionAnalysis:
    """Разобрать вопрос: сущности, тип, посылки, редакции."""
    meta = meta if meta is not None else load_doc_meta()

    entities = Entities(
        amounts=parse_amounts(question),
        inns=parse_inns(question),
        contract_numbers=parse_contract_numbers(question),
        dates=parse_dates(question),
        clauses=[m.group(1) for m in RE_CLAUSE.finditer(question)],
        articles=[m.group(1) for m in RE_ARTICLE.finditer(question)],
        counterparties=find_counterparties(question, meta),
    )

    is_dialog = detect_dialog(question)
    claims = find_claims(question, entities)
    kind = detect_kind(question, entities, meta)

    # дата события — только для вопросов про применимую редакцию
    event_date = None
    if kind == "edition_dependent" and entities.dates:
        event_date = entities.dates[0]

    preferred, superseded, edition_notes = ([], [], [])
    if kind in ("edition_dependent", "multi_source") or entities.dates:
        preferred, superseded, edition_notes = choose_editions(entities, meta, event_date)

    analysis = QuestionAnalysis(
        raw=question,
        search_text=query_prep.prepare_search_query(question),
        entities=entities,
        kind=kind,
        is_dialog=is_dialog,
        claims=claims,
        event_date=event_date,
        preferred_doc_ids=preferred,
        superseded_doc_ids=superseded,
        notes=edition_notes,
    )

    if kind == "out_of_scope":
        analysis.notes.append("в базе знаний, вероятно, нет ответа — нужен обоснованный отказ")
    if is_dialog:
        analysis.notes.append("вопрос содержит историю диалога — учитываем предыдущую реплику")
    return analysis
