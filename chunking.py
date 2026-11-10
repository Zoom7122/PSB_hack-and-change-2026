"""Шаг 2: метаданные документов и нарезка на самодостаточные фрагменты.

Зачем этот модуль существует
============================
Поиск (шаг 4) и модель (шаг 6) работают не с документами, а с фрагментами.
Качество ответа на 80% определяется тем, что попало в контекст. Поэтому:

1. Фрагмент должен быть СМЫСЛОВОЙ ЕДИНИЦЕЙ. Пункт договора 6.2 целиком, а не
   «первые 1000 символов документа». Если порезать по символам, пункт может
   разорваться на два фрагмента, поиск найдёт половину, и модель не назовёт
   «акт о выявленных недостатках» — ответ будет хуже эталона.

2. Фрагмент должен быть САМОДОСТАТОЧНЫМ. Текст «Приёмка Товара по количеству
   производится в момент передачи» бесполезен: непонятно, из какого документа,
   какой пункт, какая редакция. Поэтому к телу фрагмента мы приклеиваем «шапку»
   с реквизитами. Модель потом просто копирует её в ссылку.

3. У фрагмента должны быть МЕТАДАННЫЕ: номер документа, ИНН, дата, редакция,
   статус. Без них невозможно: отфильтровать утративший силу регламент,
   различить «Альфа-Трейд» и «Альфа-Трейдинг», подтянуть доп. соглашение.

Результат работы модуля: список Chunk + реестр DocMeta по каждому документу.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ingest import RawDoc

# ---------------------------------------------------------------------------
# РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ
# Каждое прокомментировано: регулярка без объяснения — это будущий баг.
# ---------------------------------------------------------------------------

# Номер документа: «А-08/2024», «П-17/2024», «ПТ-05/2023», «ЮО-7», «К-12», «14-П», «1»
RE_NUMBER = re.compile(r"№\s*([А-ЯЁA-Z]{0,4}[-–]?\s?\d{1,4}(?:\s?[/\-]\s?\d{2,4})?(?:-П)?)")

# Дата цифрами: 01.08.2024
RE_DATE_DOT = re.compile(r"\b(\d{1,2})\.(\d{2})\.(\d{4})\b")

# Дата словами: «1 августа 2024 г.»
MONTHS = "января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря"
RE_DATE_WORD = re.compile(rf"\b(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})", re.IGNORECASE)

# Дата после тире («г. Москва — 12 февраля 2024 г.») — так оформлены договоры
RE_DATE_AFTER_DASH = re.compile(rf"—\s*(\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}})", re.IGNORECASE)

# Пункт договора: «3.1. Арендная плата…», «6.2. При обнаружении…»
RE_CLAUSE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\.?\s+(?=\S)")

# Раздел договора: «1. Предмет договора» (короткая строка, без точки в конце)
RE_SECTION = re.compile(r"^\s*(\d{1,2})\.\s+([А-ЯЁ][^\n]{2,80})$")

# Часть статьи закона: «1. Субъект персональных данных…» (но НЕ подпункт «1)»)
RE_LAW_PART = re.compile(r"^\s*(\d{1,2})\.\s+(?=[А-ЯЁA-Z«\"(])")

# Заголовок статьи: «Статья 70. Испытание при приеме на работу»
RE_ARTICLE = re.compile(r"Статья\s+([\d.]+)\.\s*([^\n]{0,120})")

# Редакция: «Редакция от 01.03.2025»
RE_EDITION = re.compile(r"Редакция\s+от\s+(\d{2}\.\d{2}\.\d{4})", re.IGNORECASE)

# Дата выгрузки актуальной редакции закона: «Дата выгрузки: 10.10.2026»
RE_VALID_AS_OF = re.compile(r"Дата выгрузки:\s*(\d{2}\.\d{2}\.\d{4})")

# ИНН и название организации
RE_INN = re.compile(r"ИНН\s*(\d{10,12})")
RE_OOO = re.compile(r"ООО\s*«([^»]{2,60})»")

# Ссылка на родительский договор: «к договору поставки № П-17/2024»
RE_PARENT_CONTRACT = re.compile(r"к\s+договору[^№]{0,40}№\s*([А-ЯЁA-Z]{0,4}[-–]?\d{1,4}/\d{2,4})", re.IGNORECASE)

# Признак того, что в текст документа вклеена инструкция для ИИ (prompt injection).
# Мы НЕ удаляем её на этом шаге (текст документа — это данные), но помечаем чанк,
# чтобы на шаге 6 вырезать опасные предложения перед отправкой модели.
RE_INJECTION = re.compile(
    r"(игнорируй[^\n]{0,160}"
    r"|для\s+ИИ-?АССИСТЕНТА[^\n]{0,220}"
    r"|не\s+упоминай[^\n]{0,140}"
    r"|забудь\s+(?:все\s+)?(?:предыдущие\s+)?инструкции[^\n]{0,140})",
    re.IGNORECASE,
)

# Ключевые слова для определения типа документа.
# ПОРЯДОК ВАЖЕН: проверяем от самого специфичного к самому общему.
# Пример ловушки: «РЕГЛАМЕНТ СОГЛАСОВАНИЯ ДОГОВОРОВ ООО «МЕРИДИАН»» содержит
# слово «ДОГОВОР» как часть «ДОГОВОРОВ». Если проверять «договор» раньше
# «регламент», регламент будет ошибочно считаться договором — и получит
# чужие метаданные (номер, контрагента, дату).
KIND_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("доп.соглашение", ("ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ", "ДОП СОГЛАШЕНИЕ")),
    ("записка", ("СЛУЖЕБНАЯ ЗАПИСКА",)),
    ("письмо", ("ПИСЬМО",)),
    ("регламент", ("РЕГЛАМЕНТ",)),
    ("договор", ("ДОГОВОР",)),
)

# Как в ссылке называть кодекс/закон (для законов)
CODE_BY_MARKER = (
    ("КОАП", "КоАП РФ"),
    ("ТК РФ", "ТК РФ"),
    ("НК РФ", "НК РФ"),
)

# Код документа в имени файла: «А-08-2024», «П-17-2024», «К-12», «ЮО-7»
RE_DOC_CODE = re.compile(r"[А-ЯЁ]{1,3}-\d{1,3}(?:-\d{4})?")


# ---------------------------------------------------------------------------
# СТРУКТУРЫ ДАННЫХ
# ---------------------------------------------------------------------------
@dataclass
class DocMeta:
    """Всё, что мы знаем о документе и что попадёт в «шапку» каждого фрагмента."""

    doc_id: str
    group: str                      # "документы компании" | "законы"
    kind: str                       # договор / доп.соглашение / регламент / письмо / записка / закон
    citation_title: str             # как ссылаться в ответе: «Договор аренды № А-08/2024»
    number: str = ""                # «А-08/2024»
    date: str = ""                  # дата документа (дд.мм.гггг)
    edition: str = ""               # дата редакции (для регламентов)
    status: str = "действует"       # «действует» | «утратил силу»
    counterparty: str = ""          # контрагент (для договоров)
    inn: str = ""                   # ИНН контрагента
    article: str = ""               # статья закона («70»)
    code: str = ""                  # кодекс/закон («ТК РФ», «152-ФЗ»)
    valid_as_of: str = ""           # дата выгрузки действующей редакции закона
    parent_contract: str = ""       # для доп. соглашения: номер материнского договора
    has_injection: bool = False     # в документе есть вклеенная инструкция
    filename_title: str = ""        # название из имени файла (запасной вариант)


@dataclass
class Chunk:
    """Один самодостаточный фрагмент для поиска и для контекста модели."""

    chunk_id: str       # «<doc_id>::п. 6.2#0» — уникален и читаем
    doc_id: str
    label: str          # «п. 6.2» | «ч. 2» | «преамбула» | «фрагмент 3»
    section: str        # «6. Приёмка Товара» (для законов — заголовок статьи)
    body: str           # чистый текст пункта, без шапки
    prefix: str         # шапка с реквизитами документа
    text: str           # prefix + body: именно это уйдёт в индекс и в контекст
    order: int          # порядок внутри документа (нужен для «соседних пунктов»)
    kind: str = ""      # тип документа (копия из DocMeta для удобства фильтрации)
    has_injection: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# МЕЛКИЕ ПОМОЩНИКИ
# ---------------------------------------------------------------------------
def first_lines(text: str, n: int = 8) -> list[str]:
    """Первые n непустых строк — там живут реквизиты документа."""
    return [ln.strip() for ln in text.splitlines() if ln.strip()][:n]


def date_to_dots(value: str) -> str:
    """Любую дату привести к виду дд.мм.гггг, чтобы даты сравнивались строками."""
    m = RE_DATE_WORD.search(value)
    if m:
        day, month_name, year = m.groups()
        month = MONTHS.split("|").index(month_name.lower()) + 1
        return f"{int(day):02d}.{month:02d}.{year}"
    m = RE_DATE_DOT.search(value)
    if m:
        return ".".join(m.groups())
    return ""


def normalize_number(raw: str) -> str:
    """«А-08-2024» -> «А-08/2024»: убираем пробелы, дефис перед годом делаем слэшем."""
    num = re.sub(r"\s+", "", raw).replace("–", "-")
    return re.sub(r"-(\d{4})$", r"/\1", num)


def detect_kind(doc: RawDoc, lines: list[str]) -> str:
    """Тип документа: законы — по группе, остальные — по первым строкам.

    Если в тексте ключевых слов нет (например, письмо начинается с названия
    организации), смотрим на имя файла: организаторы назвали файлы осмысленно.
    """
    if doc.group == "законы":
        return "закон"

    probe = " ".join(lines[:3]).upper()
    for kind, markers in KIND_ORDER:
        if any(marker in probe for marker in markers):
            return kind

    name_probe = doc.name.upper().replace("_", " ")
    for kind, markers in KIND_ORDER:
        if any(marker in name_probe for marker in markers):
            return kind
    return "прочее"


# ---------------------------------------------------------------------------
# МЕТАДАННЫЕ ДОКУМЕНТА
# ---------------------------------------------------------------------------
def extract_doc_meta(doc: RawDoc) -> DocMeta:
    """Достать из документа всё, что понадобится поиску, ссылкам и проверкам."""
    lines = first_lines(doc.text, 10)
    head = "\n".join(lines[:6])
    kind = detect_kind(doc, lines)

    meta = DocMeta(
        doc_id=doc.doc_id,
        group=doc.group,
        kind=kind,
        citation_title="",           # заполним ниже
        filename_title=doc.title,
        has_injection=bool(RE_INJECTION.search(doc.text)),
    )

    # --- статус: «утратил силу» только про САМ документ ---------------------
    # Ловушка: в шапке действующего регламента 2025 написано
    # «Редакция от 10.01.2023 утрачивает силу» — это про другую редакцию.
    # Поэтому смотрим только ПЕРВУЮ строку и только на явную конструкцию
    # «РЕДАКЦИЯ УТРАТИЛА СИЛУ …» в её начале, плюс подсказку в имени файла.
    first = lines[0] if lines else ""
    if re.match(r"^\s*РЕДАКЦИЯ\s+УТРАТИЛА\s+СИЛУ", first, re.IGNORECASE) or "утратил силу" in doc.title.lower():
        meta.status = "утратил силу"

    if kind == "закон":
        _fill_law_meta(meta, head, doc)
    else:
        _fill_company_meta(meta, doc, lines, head)

    meta.citation_title = build_citation_title(meta, lines)
    return meta


def _fill_law_meta(meta: DocMeta, head: str, doc: RawDoc) -> None:
    """Кодекс/закон, номер статьи, дата выгрузки действующей редакции."""
    m = RE_ARTICLE.search(head)
    if m:
        meta.article = m.group(1).rstrip(".")

    upper = head.upper()
    for marker, code in CODE_BY_MARKER:
        if marker in upper:
            meta.code = code
            break
    if not meta.code:
        m = re.search(r"(\d{2,3}-ФЗ)", head)          # 152-ФЗ, 82-ФЗ
        meta.code = m.group(1) if m else doc.title.split()[0]

    m = RE_VALID_AS_OF.search(head)
    if m:
        # Для законов «дата» — это дата, на которую редакция действует.
        # Именно её нельзя использовать как «дату принятия»: она всегда больше
        # любой даты из вопроса, и любой фильтр по ней выкинет все нормы.
        meta.valid_as_of = m.group(1)
    meta.status = "действует"                          # в базе только действующие редакции


def _fill_company_meta(meta: DocMeta, doc: RawDoc, lines: list[str], head: str) -> None:
    """Номер, дата, редакция, контрагент, ИНН, материнский договор."""
    # Номер документа ищем в первых строках, но у регламента «№ 14-П» — это
    # номер ПРИКАЗА, которым регламент утверждён, а не номер регламента. Поэтому
    # для регламентов номер не заполняем.
    if meta.kind not in {"регламент"}:
        for line in lines[:4]:
            m = RE_NUMBER.search(line)
            if m:
                meta.number = normalize_number(m.group(1))
                break

    # Дата документа. Порядок поиска важен:
    # 1) «… от 15.04.2025» — так оформлены письмо и служебная записка;
    # 2) «г. Москва — 12 февраля 2024 г.» — так оформлены договоры;
    # 3) любая дата словом/цифрами в первых строках — запасной вариант.
    # У регламента дату НЕ берём: «от 17.02.2025» — это дата приказа об
    # утверждении, а не дата документа. Для регламента важна редакция.
    if meta.kind == "регламент":
        lines_for_date: list[str] = []
    else:
        lines_for_date = lines
    for line in lines_for_date[:6]:
        m = re.search(r"от\s+(\d{2}\.\d{2}\.\d{4})", line)
        if m:
            meta.date = m.group(1)
            break
    if not meta.date and meta.kind != "регламент":
        m = RE_DATE_AFTER_DASH.search(head)
        if m:
            meta.date = date_to_dots(m.group(1))
    if not meta.date and meta.kind != "регламент":
        for line in lines_for_date[:6]:
            value = date_to_dots(line)
            if value:
                meta.date = value
                break

    # Редакция (для регламентов) — «Редакция от 01.03.2025»
    m = RE_EDITION.search(head)
    if m:
        meta.edition = m.group(1)

    # Контрагент и ИНН — только для договоров (первое упоминание ООО = поставщик/арендодатель)
    if meta.kind in {"договор", "доп.соглашение"}:
        m = RE_OOO.search(doc.text[:800])
        if m:
            meta.counterparty = m.group(1)
        m = RE_INN.search(doc.text[:1500])
        if m:
            meta.inn = m.group(1)

    # Родительский договор — для доп. соглашения
    if meta.kind == "доп.соглашение":
        m = RE_PARENT_CONTRACT.search(head)
        if m:
            meta.parent_contract = normalize_number(m.group(1))


def build_citation_title(meta: DocMeta, lines: list[str]) -> str:
    """Собрать короткое юридически корректное название для ссылки в ответе.

    Именно эту строку модель будет копировать в «Источник: …», поэтому она
    должна совпадать с тем, как документ называют в реальности.
    """
    # --- законы: «ТК РФ, ст. 70» -------------------------------------------
    if meta.kind == "закон":
        return f"{meta.code}, ст. {meta.article}".strip(", ")

    # --- регламенты: имя + редакция + статус -------------------------------
    if meta.kind == "регламент":
        title = "Регламент согласования договоров ООО «Меридиан»"
        if meta.edition:
            title += f" (ред. от {meta.edition}"
            title += ", утратил силу)" if meta.status != "действует" else ")"
        return title

    # --- доп. соглашение: своё название + ссылка на материнский договор -----
    if meta.kind == "доп.соглашение":
        title = f"Дополнительное соглашение № {meta.number or '1'}"
        if meta.parent_contract:
            title += f" к договору поставки № {meta.parent_contract}"
        return title

    # --- остальные документы компании ---------------------------------------
    # Основу берём из ИМЕНИ ФАЙЛА: организаторы назвали файлы осмысленно и в
    # нормальном регистре («Письмо комплаенс-службы К-12 от 15.04.2025»).
    # Тело документа для этого не годится: у письма первая строка — «ООО «МЕРИДИАН»»,
    # а у регламента первой строкой идёт пометка об утрате силы.
    base = re.sub(r"\s*\([^)]*утратил[^)]*\)", "", meta.filename_title).strip()

    # Ставим «№» перед кодом документа и приводим код к каноничному виду:
    # «Договор аренды А-08-2024 Северный Склад» -> «Договор аренды № А-08/2024 …»
    code = RE_DOC_CODE.search(base)
    if code and "№" not in base:
        base = base.replace(code.group(0), f"№ {normalize_number(code.group(0))}", 1)
    elif meta.number and meta.number not in base:
        base = f"{base} № {meta.number}"

    # Письму и записке добавляем дату: она есть в теле («Исх. № К-12 от 15.04.2025»,
    # «СЛУЖЕБНАЯ ЗАПИСКА № ЮО-7 от 12.05.2025»), но может отсутствовать в имени файла.
    if meta.kind in {"письмо", "записка"} and meta.date and "от " not in base:
        base = f"{base} от {meta.date}"
    return base


# ---------------------------------------------------------------------------
# НАРЕЗКА: вспомогательные функции
# ---------------------------------------------------------------------------
def split_long(text: str, max_chars: int) -> list[str]:
    """Разрезать слишком длинный кусок по границам абзацев, потом предложений.

    Зачем: в базе есть части статей по 5–8 тыс. символов (НК ст. 164 — 79 тыс.
    целиком). Если отдать такой кусок одним фрагментом, он займёт весь контекст
    и вытеснит остальные нужные фрагменты.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    buffer = ""
    for paragraph in text.split("\n"):
        # Отделяем ведущий номер («4.», «2.1.») перед разбиением на предложения.
        # Зачем: сплиттер по «. » иначе превратит «4. При получении…» в два
        # куска — «4.» и «При получении…». Тогда первый фрагмент статьи будет
        # состоять из двух символов и навсегда засорит поиск.
        lead = ""
        m = re.match(r"^(\d{1,2}(?:\.\d{1,2})*(?:-\d)?\.)\s+", paragraph)
        if m:
            lead = m.group(1) + " "
            paragraph = paragraph[m.end():]

        candidate = f"{buffer}\n{paragraph}" if buffer else paragraph
        if len(candidate) <= max_chars:
            buffer = candidate
            continue
        if buffer:
            pieces.append(buffer)
        # Абзац сам длиннее лимита — режем по предложениям, затем по символам
        buffer = ""
        sentences = re.split(r"(?<=[.;:])\s+", paragraph)
        if sentences and lead:
            sentences[0] = lead + sentences[0]
        for sentence in sentences:
            if len(sentence) > max_chars:
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                pieces.extend(sentence[i:i + max_chars] for i in range(0, len(sentence), max_chars))
                continue
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                pieces.append(buffer)
                buffer = sentence
    if buffer:
        pieces.append(buffer)
    return [p.strip() for p in pieces if p.strip()]


def group_paragraphs(lines: list[str], target_chars: int = 900) -> list[str]:
    """Сгруппировать абзацы в куски по ~target_chars.

    Зачем отдельная функция: не все документы пронумерованы. Письмо комплаенс и
    служебная записка — сплошной текст; ТК ст. 70 в этой выгрузке тоже без
    номеров частей. Резать их «по 1000 символов» нельзя (разорвём предложение),
    а оставлять целиком — плохо для поиска: нужный абзац утонет в остальном.
    Поэтому группируем по абзацам, не разрывая их.
    """
    groups: list[str] = []
    buffer: list[str] = []
    size = 0
    for line in lines:
        if not line.strip():
            continue
        if size + len(line) > target_chars and buffer:
            groups.append("\n".join(buffer))
            buffer, size = [], 0
        buffer.append(line)
        size += len(line) + 1
    if buffer:
        groups.append("\n".join(buffer))
    return groups


# ---------------------------------------------------------------------------
# НАРЕЗКА: документы компании (договоры, регламенты, письма, записки)
# ---------------------------------------------------------------------------
def company_units(text: str) -> list[tuple[str, str, str]]:
    """Разбить документ компании на (label, section, body).

    Правила:
      * «1. Предмет договора» — это РАЗДЕЛ. Он НЕ становится отдельным фрагментом
        (иначе получим чанк из 19 символов, который ничего не значит), а
        превращается в подпись `section` для всех пунктов этого раздела;
      * «3.1. Арендная плата…» — ПУНКТ, он и есть фрагмент;
      * всё до первого раздела/пункта — ПРЕАМБУЛА (стороны, ИНН, предмет).
        Это обязательный отдельный фрагмент: по нему ищут контрагента и ИНН;
      * если нумерованных пунктов нет вообще — группируем абзацы.
    """
    lines = text.split("\n")
    has_clauses = any(RE_CLAUSE.match(line.strip()) for line in lines if line.strip())
    if not has_clauses:
        groups = group_paragraphs(lines)
        if len(groups) == 1:
            return [("текст", "", groups[0])]
        return [(f"фрагмент {i}", "", g) for i, g in enumerate(groups, 1)]

    units: list[tuple[str, str, str]] = []
    section = ""
    label = "преамбула"
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            units.append((label, section, body))

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if buffer and buffer[-1] != "":
                buffer.append("")
            continue

        m_section = RE_SECTION.match(line)
        m_clause = RE_CLAUSE.match(line)

        # Раздел: закрываем предыдущий кусок и меняем подпись, саму строку
        # в текст не кладём — она уйдёт в `section`.
        if m_section and not line.endswith("."):
            flush()
            buffer = []
            section = f"{m_section.group(1)}. {m_section.group(2).strip()}"
            label = f"раздел {m_section.group(1)}"
            continue

        if m_clause:
            flush()
            buffer = []
            label = f"п. {m_clause.group(1)}"
            # Обрезаем номер из начала строки: иначе получим «п. 5.3. 5.3. Стороны…»
            buffer.append(line[m_clause.end():].strip())
            continue

        buffer.append(line)

    flush()
    return units


# ---------------------------------------------------------------------------
# НАРЕЗКА: статьи законов
# ---------------------------------------------------------------------------
def find_article_heading(lines: list[str]) -> tuple[int, str]:
    """Найти строку-заголовок «Статья N. Название» и вернуть (индекс, заголовок).

    Тонкость, из-за которой это отдельная функция: слово «Статья» встречается
    в первой строке файла внутри служебной пометки «Источник: Статья 70. ТК РФ
    Испытание при приеме на работу. Трудовой кодекс …». Если взять её за
    заголовок, то (1) в заголовок попадёт весь мусор до конца строки, а
    (2) тело статьи начнётся со строки «URL: https://…» и «Дата выгрузки: …» —
    и эти служебные строки попадут в индекс как будто это нормы права.

    Поэтому сначала ищем настоящий заголовок: строку, которая НАЧИНАЕТСЯ с
    «Статья N.» и не является служебной. И только если такого нет — берём
    заголовок из пометки «Источник:».
    """
    service_prefixes = ("Источник:", "URL:", "Дата выгрузки:")

    for i, line in enumerate(lines[:80]):
        stripped = line.strip()
        if stripped.startswith(service_prefixes):
            continue
        m = re.match(r"^(Статья\s+[\d.]+\.[^\n]{0,140})", stripped)
        if m:
            title = re.sub(r"\s{2,}", " ", m.group(1)).strip()
            return i + 1, title

    for i, line in enumerate(lines[:20]):
        m = RE_ARTICLE.search(line)
        if m:
            return i + 1, f"Статья {m.group(1)}. {m.group(2).strip()}".strip(" .")
    return 0, ""


def law_units(text: str) -> tuple[list[tuple[str, str, str]], str]:
    """Разбить статью закона на (label, section, body) + вернуть заголовок статьи.

    Два режима:
      1) В статье есть нумерованные части («1. …», «2. …») — один фрагмент
         на часть. Подпункты («1)», «2.1)») остаются внутри своей части.
      2) Частей нет (так выгружены некоторые статьи ТК) — группируем абзацы.

    Шапка файла («Источник: …», «URL: …», «Часть 1 изменена с …») отбрасывается:
    это служебные пометки, а не нормы. Дата выгрузки уже сохранена в метаданных.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    body_start, article_title = find_article_heading(lines)
    body = lines[body_start:]

    part_indices = [i for i, line in enumerate(body) if RE_LAW_PART.match(line.strip())]

    # --- режим 1: есть нумерованные части -----------------------------------
    if len(part_indices) >= 2:
        units: list[tuple[str, str, str]] = []
        label, buffer = "текст статьи", []
        notes_mode = False   # после строки «Примечания» нумерация начинается заново

        def flush() -> None:
            chunk_body = "\n".join(buffer).strip()
            if chunk_body:
                units.append((label, article_title, chunk_body))

        for line in body:
            stripped = line.strip()
            if not stripped:
                if buffer and buffer[-1] != "":
                    buffer.append("")
                continue
            # «Примечания» в статьях КоАП/НК нумеруются отдельно: если этого не
            # учесть, в документе окажутся две «ч. 1» и две «ч. 4», и ссылка
            # «ч. 4» станет неоднозначной.
            if re.match(r"^Примечани", stripped, re.IGNORECASE):
                flush()
                buffer = [stripped]
                label = "примечания"
                notes_mode = True
                continue

            m_part = RE_LAW_PART.match(stripped)
            if m_part:
                flush()
                buffer = []
                label = f"{'прим.' if notes_mode else 'ч.'} {m_part.group(1)}"
                # Номер из начала строки убираем: подпись «ч. 2» уже содержит его,
                # иначе получится «ч. 2. 2. Согласие на обработку…».
                buffer.append(stripped[m_part.end():])
                continue
            buffer.append(stripped)
        flush()
        return _merge_tiny(units), article_title

    # --- режим 2: частей нет — группируем абзацы ----------------------------
    groups = group_paragraphs(body)
    if len(groups) <= 1:
        return [("текст", article_title, "\n".join(groups))], article_title
    return [(f"фрагмент {i}", article_title, g) for i, g in enumerate(groups, 1)], article_title


def _merge_tiny(units: list[tuple[str, str, str]], threshold: int = 80) -> list[tuple[str, str, str]]:
    """Приклеить совсем короткие куски к предыдущему.

    В законах между частями встречаются строки вида «Часть вторая утратила силу.»
    Отдельный фрагмент из 30 символов только засоряет поиск.
    """
    merged: list[tuple[str, str, str]] = []
    for label, section, body in units:
        if merged and len(body) < threshold:
            prev_label, prev_section, prev_body = merged[-1]
            merged[-1] = (prev_label, prev_section, f"{prev_body}\n{body}")
        else:
            merged.append((label, section, body))
    return merged


# ---------------------------------------------------------------------------
# СБОРКА ФРАГМЕНТА
# ---------------------------------------------------------------------------
def build_prefix(meta: DocMeta, section: str) -> str:
    """Шапка фрагмента: реквизиты документа + раздел.

    Пример результата:
        Договор поставки № П-17/2024 | ООО «Альфа-Трейд», ИНН 7701234567, от 12.02.2024
        6. Приёмка Товара
    """
    parts = [meta.citation_title]
    extras: list[str] = []
    if meta.counterparty:
        extras.append(f"ООО «{meta.counterparty}»")
    if meta.inn:
        extras.append(f"ИНН {meta.inn}")
    if meta.date:
        extras.append(f"от {meta.date}")
    if meta.valid_as_of:
        extras.append(f"действующая редакция на {meta.valid_as_of}")
    if meta.status != "действует":
        extras.append("РЕДАКЦИЯ УТРАТИЛА СИЛУ")
    if extras:
        parts[0] = parts[0] + " | " + ", ".join(extras)
    if section:
        parts.append(section)
    return "\n".join(parts)


def chunk_document(doc: RawDoc, meta: DocMeta, max_chars: int = 1400) -> list[Chunk]:
    """Нарезать один документ на фрагменты."""
    if meta.kind == "закон":
        # Для законов `section` — это заголовок статьи: «Статья 70. Испытание…».
        # Он короткий (одна строка), поэтому его не страшно повторять в шапке
        # каждого фрагмента: в отличие от всей шапки файла он не занимает места.
        units, article_title = law_units(doc.text)
        section_default = article_title
    else:
        units = company_units(doc.text)
        section_default = ""

    chunks: list[Chunk] = []
    order = 0
    for label, section, body in units:
        section = section or section_default
        for piece_index, piece in enumerate(split_long(body, max_chars)):
            label_full = label if piece_index == 0 else f"{label} (продолжение {piece_index})"
            prefix = build_prefix(meta, section)
            # Собираем текст фрагмента: шапка + подпись + тело.
            # Подпись («п. 6.2.») нужна, чтобы модель видела, на что ссылаться.
            if label_full.startswith(("п.", "ч.", "раздел", "прим.")):
                head_line = f"{label_full}. {piece}"
            else:
                head_line = f"{label_full} {piece}"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{label_full}#{order}",
                    doc_id=doc.doc_id,
                    label=label_full,
                    section=section,
                    body=piece,
                    prefix=prefix,
                    text=f"{prefix}\n{head_line.strip()}",
                    order=order,
                    kind=meta.kind,
                    has_injection=bool(RE_INJECTION.search(piece)),
                )
            )
            order += 1
    return chunks


def build_chunks(docs: list[RawDoc], max_chars: int = 1400, verbose: bool = True) -> tuple[list[Chunk], dict[str, DocMeta]]:
    """Нарезать весь корпус. Возвращает фрагменты и реестр метаданных."""
    all_chunks: list[Chunk] = []
    metas: dict[str, DocMeta] = {}
    for doc in docs:
        meta = extract_doc_meta(doc)
        chunks = chunk_document(doc, meta, max_chars=max_chars)
        metas[doc.doc_id] = meta
        all_chunks.extend(chunks)
        if verbose:
            print(f"[chunk] {len(chunks):3d} | {meta.status:<13} | {meta.citation_title[:60]}")
    return all_chunks, metas
