"""Подготовка доказательной базы: что из пришедших чанков можно показать модели.

Шаги (все детерминированные, без LLM):
  1. дедупликация по chunk_id и проверка corpus_version;
  2. вычистка попыток управлять моделью из текста документов (prompt injection);
  3. определение периода события (event_date или дата из вопроса);
  4. применимость редакции каждого документа к этому периоду —
     по effective_from/effective_to, а если их нет — по тексту документа
     («утратила силу с …», «действует с …», «изменена с …»);
  5. сверка чисел из вопроса с документами (ловушки «почему 500 000?»).

Результат — пронумерованный список Evidence. Номер [n] — единственное, чем
модель ссылается на источник: chunk_id она не видит и не может выдумать.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import date

from .contracts import GenerationRequest, RerankedChunk
from .numbers import MONTHS_GENITIVE, amount_like, numeric_facts, parse_dates

# --------------------------------------------------------------- инъекции
INJECTION_PATTERNS = [
    re.compile(r"внимани[ея]\s+(?:для|к)\s+(?:ии|ai|ассистент|модел|нейросет|бот)", re.I),
    re.compile(r"(?:для|к)\s+(?:ии|ai)[\s-]*(?:ассистент|модел|бот)", re.I),
    re.compile(r"игнорир\w*\s+(?:все\s+|любые\s+)?(?:остальн|предыдущ|прежн|други|эт|систем|инструкц|документ|правил|указан)", re.I),
    re.compile(r"(?:забудь|отмени)\w*\s+(?:все\s+)?(?:предыдущ|прежн|инструкц|правил)", re.I),
    re.compile(r"не\s+упоминай\w*\s+(?:эту\s+|данную\s+)?(?:инструкц|указан|правил)", re.I),
    re.compile(r"на\s+(?:любой|все|каждый)\s+вопрос\w*\s+(?:\S+\s+){0,10}отвечай", re.I),
    re.compile(r"(?:системн\w+\s+(?:промпт|инструкц|сообщени)|system\s+prompt)", re.I),
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|other)", re.I),
    re.compile(r"\byou\s+(?:are\s+now|must)\b", re.I),
    re.compile(r"\bты\s+(?:теперь|должен|обязан)\b", re.I),
    re.compile(r"</?\s*(?:system|assistant|fragment|instructions?)\s*>", re.I),
]
RE_SENTENCE = re.compile(r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)|\n")


def sanitize_text(text: str) -> tuple[str, bool]:
    """Вырезает предложения с попытками управлять моделью. → (текст, было ли)."""
    removed = False
    kept: list[str] = []
    for sentence in RE_SENTENCE.findall(text):
        if any(pattern.search(sentence) for pattern in INJECTION_PATTERNS):
            removed = True
            continue
        kept.append(sentence)
    clean = "".join(kept)
    clean = re.sub(r"[ \t]+\n", "\n", clean)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return clean, removed


# ------------------------------------------------------------ период события
@dataclass(frozen=True)
class EventPeriod:
    start: date
    end: date
    source: str  # "request" | "question"

    def describe(self) -> str:
        if self.start == self.end:
            return self.start.strftime("%d.%m.%Y")
        return f"{self.start:%d.%m.%Y}–{self.end:%d.%m.%Y}"


_MONTHS_PREPOSITIONAL = {
    "январе": 1, "феврале": 2, "марте": 3, "апреле": 4, "мае": 5, "июне": 6,
    "июле": 7, "августе": 8, "сентябре": 9, "октябре": 10, "ноябре": 11, "декабре": 12,
}
_MONTHS_NOMINATIVE = {
    "январь": 1, "февраль": 2, "март": 3, "апрель": 4, "май": 5, "июнь": 6,
    "июль": 7, "август": 8, "сентябрь": 9, "октябрь": 10, "ноябрь": 11, "декабрь": 12,
}
_MONTH_FORMS = {**_MONTHS_PREPOSITIONAL, **_MONTHS_NOMINATIVE, **MONTHS_GENITIVE}
RE_MONTH_YEAR = re.compile(
    rf"(?<!\d\s)\b({'|'.join(_MONTH_FORMS)})\s+(\d{{4}})", re.I)
RE_YEAR = re.compile(r"(?<![\d/\-.])\b(\d{4})\s*(?:году|года|г\.)", re.I)


def infer_event_period(question: str) -> EventPeriod | None:
    """Дата события из текста вопроса. Неоднозначность → None.

    Примеры: «01.08.2024» → день; «в августе 2024 года» → месяц;
    «в 2024 году» → год. Две разные даты — это не «дата события», а
    условие задачи; решать за модель не будем.
    """
    periods: set[tuple[date, date]] = set()
    rest = question
    for day in parse_dates(question):
        periods.add((day, day))
    rest = re.sub(r"\d{1,2}\.\d{1,2}\.\d{4}", " ", rest)
    rest = re.sub(rf"\d{{1,2}}\s+(?:{'|'.join(MONTHS_GENITIVE)})\s+\d{{4}}", " ", rest, flags=re.I)
    for month_name, year in RE_MONTH_YEAR.findall(rest):
        month, year_num = _MONTH_FORMS[month_name.lower()], int(year)
        last = calendar.monthrange(year_num, month)[1]
        periods.add((date(year_num, month, 1), date(year_num, month, last)))
    rest = RE_MONTH_YEAR.sub(" ", rest)
    for year in RE_YEAR.findall(rest):
        periods.add((date(int(year), 1, 1), date(int(year), 12, 31)))
    if len(periods) != 1:
        return None
    start, end = periods.pop()
    if not 1990 <= start.year <= 2100:
        return None
    return EventPeriod(start, end, "question")


# ------------------------------------------------------- редакции документов
_DATE = rf"(\d{{1,2}}\.\d{{1,2}}\.\d{{4}}|\d{{1,2}}\s+(?:{'|'.join(MONTHS_GENITIVE)})\s+\d{{4}})"
RE_EXPIRED = re.compile(rf"утратил\w*\s+силу(?:\s+с\s+{_DATE})?", re.I)
RE_IN_FORCE = re.compile(rf"(?:действует|вступа\w+\s+в\s+силу|применяется)\s+с\s+{_DATE}", re.I)
RE_EDITION = re.compile(rf"редакци\w+\s+от\s+{_DATE}", re.I)
RE_LAW_CHANGE = re.compile(
    r"((?:Стать\w+|Част\w+|Пункт\w*|Подпункт\w*|Абзац\w*)[^.\n]{0,40}?)\s+"
    rf"(?:изменен\w*|утратил\w*\s+силу|дополнен\w*|изложен\w*[^.\n]{{0,30}}?)\s+с\s+{_DATE}",
    re.I,
)


def _first_date(text: str) -> date | None:
    dates = parse_dates(text)
    return dates[0] if dates else None


@dataclass
class DocumentEdition:
    effective_from: date | None = None
    effective_to: date | None = None      # первый день, когда редакция уже НЕ действует
    marked_expired: bool = False           # «утратил силу» без даты
    law_changes: list[tuple[str, date]] = field(default_factory=list)


def _is_law(chunk: RerankedChunk) -> bool:
    kind = (chunk.document_type or "").lower()
    if kind in {"law", "закон", "законы", "federal_law", "code", "кодекс"}:
        return True
    title = f"{chunk.document_title or ''} {chunk.document_id or ''}".lower()
    return bool(re.search(r"\b(?:нк|тк|гк|коап|ук)\s+рф\b|\d+-фз|федеральн\w+\s+закон|кодекс", title))


def detect_edition(chunks: list[RerankedChunk]) -> DocumentEdition:
    """Реквизиты редакции одного документа по метаданным и тексту всех его чанков."""
    edition = DocumentEdition()
    meta_from = next((c.effective_from for c in chunks if c.effective_from), None)
    meta_to = next((c.effective_to for c in chunks if c.effective_to), None)
    head = " \n".join(f"{c.document_title or ''}\n{c.text}" for c in chunks)

    is_law = any(_is_law(c) for c in chunks)
    # У законов «утратил силу»/«действует с» относятся к отдельным частям,
    # а не к документу — их разбирает RE_LAW_CHANGE.
    expired = [] if is_law else list(RE_EXPIRED.finditer(head))
    if expired:
        # «(утратил силу)» в названии бывает без даты — берём первое упоминание с датой.
        edition.marked_expired = True
        edition.effective_to = next(
            (_first_date(m.group(1)) for m in expired if m.group(1)), None)
    in_force = None if is_law else RE_IN_FORCE.search(head)
    if in_force:
        edition.effective_from = _first_date(in_force.group(1))
    if edition.effective_from is None and not is_law:
        edition_date = RE_EDITION.search(head)
        if edition_date:
            edition.effective_from = _first_date(edition_date.group(1))
    if is_law:
        seen = set()
        for scope, raw in RE_LAW_CHANGE.findall(head):
            when = _first_date(raw)
            key = (scope.strip().lower(), when)
            if when and key not in seen:
                seen.add(key)
                edition.law_changes.append((scope.strip(), when))
    # Метаданные хранилища надёжнее текста — они перекрывают.
    edition.effective_from = meta_from or edition.effective_from
    edition.effective_to = meta_to or edition.effective_to
    return edition


def applicability(edition: DocumentEdition, period: EventPeriod | None,
                  is_law: bool, today: date) -> tuple[str, str]:
    """→ (код, пояснение для модели).

    Коды: applicable, not_yet_in_force, expired, ambiguous, superseded, current, unknown.
    """
    start, end = edition.effective_from, edition.effective_to
    fmt = lambda d: d.strftime("%d.%m.%Y")  # noqa: E731

    if is_law:
        notes = ["действующая редакция закона (иных редакций в базе нет)"]
        future = [(scope, when) for scope, when in edition.law_changes if when > today]
        if future and not period:
            changes = "; ".join(f"{scope} — с {fmt(when)}" for scope, when in future[:4])
            notes.append(f"часть изменений ещё не вступила в силу ({changes})")
            return "ambiguous", "; ".join(notes)
        if period and edition.law_changes:
            later = [(scope, when) for scope, when in edition.law_changes if when > period.start]
            if later:
                changes = "; ".join(f"{scope} — изменения с {fmt(when)}" for scope, when in later[:4])
                notes.append(
                    f"после даты события вносились изменения ({changes}); редакции, "
                    f"действовавшей на {period.describe()}, в базе нет — выводы о ней не подтверждены")
                return "ambiguous", "; ".join(notes)
        return ("current" if not period else "applicable"), "; ".join(notes)

    if period is None:
        if edition.marked_expired or (end and end <= today):
            since = f" с {fmt(end)}" if end else ""
            return "superseded", f"редакция утратила силу{since}; к текущим ситуациям не применяется"
        if start and start > today:
            return "not_yet_in_force", f"редакция вступает в силу с {fmt(start)}"
        if start:
            return "current", f"действующая редакция (с {fmt(start)})"
        return "unknown", "сведений о редакции нет"

    if start and period.end < start:
        return "not_yet_in_force", (
            f"редакция действует с {fmt(start)} — на дату события ({period.describe()}) ещё не действовала")
    if end and period.start >= end:
        return "expired", (
            f"редакция утратила силу с {fmt(end)} — на дату события ({period.describe()}) уже не действовала")
    if (start and period.start < start <= period.end) or (end and period.start < end <= period.end):
        return "ambiguous", (
            f"редакция сменилась внутри периода события ({period.describe()}); уточните точную дату")
    if edition.marked_expired and end is None:
        return "ambiguous", "редакция утратила силу, дата утраты не указана"
    if start or end:
        period_text = f"с {fmt(start)}" if start else ""
        period_text += f" по {fmt(end)}" if end else ""
        return "applicable", f"редакция действовала на дату события ({period_text.strip()})"
    return "unknown", "сведений о редакции нет"


# ---------------------------------------------------------------- Evidence
@dataclass
class Evidence:
    index: int
    chunk: RerankedChunk
    text: str
    label: str
    kind: str                      # "закон" | "договор" | "внутренний документ" | "документ"
    applicability: str
    applicability_note: str
    injection_removed: bool = False

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    def haystack(self) -> str:
        """Всё, где может законно встретиться число из утверждения."""
        c = self.chunk
        parts = [self.text, self.label, c.document_title or "", c.section or "",
                 c.document_number or "", str(c.extra_field("cite_label") or "")]
        return "\n".join(parts)


@dataclass
class PreparedEvidence:
    items: list[Evidence]
    period: EventPeriod | None
    excluded: list[tuple[str, str]] = field(default_factory=list)   # (chunk_id, причина)
    notes: list[str] = field(default_factory=list)
    premise_hints: list[str] = field(default_factory=list)

    def by_index(self) -> dict[int, Evidence]:
        return {item.index: item for item in self.items}


def _kind(chunk: RerankedChunk) -> str:
    if _is_law(chunk):
        return "закон"
    raw = f"{chunk.document_type or ''} {chunk.document_title or ''}".lower()
    if "contract" in raw or "договор" in raw or "соглашени" in raw:
        return "договор"
    if raw.strip():
        return "внутренний документ"
    return "документ"


def _label(chunk: RerankedChunk) -> str:
    cite = chunk.extra_field("cite_label")
    title = (chunk.document_title or "").strip() or (chunk.document_id or "").strip() or "Документ"
    section = (chunk.section or "").strip()
    if section and not re.match(r"(?:п|ч|ст|абз|разд)\.?\s", section, re.I) and re.match(r"\d", section):
        section = f"п. {section}"
    base = title
    if cite and str(cite).strip() and str(cite).strip().lower() not in title.lower():
        base = f"{title} ({str(cite).strip()})"
    if chunk.revision_id:
        base += f", редакция {chunk.revision_id}"
    return f"{base}, {section}" if section and section.lower() != "текст" else base


def _corpus_filter(request: GenerationRequest) -> tuple[list[RerankedChunk], list[tuple[str, str]]]:
    """Дубликаты и чанки чужой версии корпуса.

    Версию сверяем, только если она где-то явно видна: поле corpus_version у
    чанка либо префикс chunk_id вида «<corpus_version>:…». Префикс проверяем,
    только когда схема используется (хотя бы один chunk_id начинается с версии),
    иначе любая иная схема id отрезала бы всю выдачу.
    """
    version = request.corpus_version.strip()
    prefix_scheme = bool(version) and any(
        c.chunk_id.startswith(f"{version}:") for c in request.chunks)
    kept: list[RerankedChunk] = []
    excluded: list[tuple[str, str]] = []
    seen: set[str] = set()
    for chunk in request.chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        own_version = chunk.extra_field("corpus_version")
        if version and own_version and str(own_version) != version:
            excluded.append((chunk.chunk_id, f"corpus_version {own_version} ≠ {version}"))
            continue
        if prefix_scheme and not chunk.chunk_id.startswith(f"{version}:"):
            excluded.append((chunk.chunk_id, f"chunk_id не принадлежит corpus_version {version}"))
            continue
        if not chunk.text.strip():
            excluded.append((chunk.chunk_id, "пустой текст"))
            continue
        kept.append(chunk)
    return kept, excluded


def _premise_hints(question: str, items: list[Evidence]) -> list[str]:
    """Числа из вопроса, которых нет во фрагментах, — кандидаты в ложные посылки."""
    asked = amount_like(question)
    if not asked or not items:
        return []
    have = set()
    for item in items:
        have |= numeric_facts(item.text, for_haystack=True)
    hints = []
    for fact in sorted(asked - have):
        if fact.startswith("D:"):
            continue
        try:
            value = float(fact)
        except ValueError:
            continue
        near = sorted(
            {h for h in have if re.fullmatch(r"\d+(?:\.\d+)?", h)
             and h != fact and value and 0.1 * value <= float(h) <= 10 * value},
            key=lambda h: abs(float(h) - value))[:3]
        shown = _pretty(fact)
        if near:
            hints.append(f"значение «{shown}» из вопроса во фрагментах не встречается; "
                         f"во фрагментах есть: {', '.join(_pretty(n) for n in near)}")
        else:
            hints.append(f"значение «{shown}» из вопроса во фрагментах не встречается")
    return hints[:5]


def _pretty(fact: str) -> str:
    if fact.isdigit() and len(fact) > 4:
        return f"{int(fact):,}".replace(",", " ")
    return fact


def prepare_evidence(request: GenerationRequest, *, max_chunks: int, max_chars: int,
                     infer_event_date: bool, today: date | None = None) -> PreparedEvidence:
    today = today or date.today()
    chunks, excluded = _corpus_filter(request)
    notes: list[str] = []

    period: EventPeriod | None = None
    if request.event_date:
        period = EventPeriod(request.event_date, request.event_date, "request")
    elif infer_event_date:
        period = infer_event_period(request.question)
        if period:
            notes.append(f"дата события определена по вопросу: {period.describe()}")

    # Редакцию определяем по ВСЕМ пришедшим чанкам документа: «шапка» с
    # «утратил силу» может прийти отдельным чанком, который не попадёт в лимит.
    by_document: dict[str, list[RerankedChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id or chunk.document_title or chunk.chunk_id, []).append(chunk)
    editions = {key: detect_edition(group) for key, group in by_document.items()}

    items: list[Evidence] = []
    for chunk in chunks:
        if len(items) >= max_chunks:
            excluded.append((chunk.chunk_id, "превышен лимит контекста"))
            continue
        text, removed = sanitize_text(chunk.text)
        if removed:
            notes.append(f"из фрагмента {chunk.chunk_id} удалены встроенные инструкции для ИИ")
        if not text:
            excluded.append((chunk.chunk_id, "фрагмент состоял только из инструкций для ИИ"))
            continue
        if len(text) > max_chars:
            cut = text[:max_chars]
            stop = max(cut.rfind(". "), cut.rfind("\n"))
            text = (cut[:stop + 1] if stop > max_chars // 2 else cut).rstrip() + " […]"
        key = chunk.document_id or chunk.document_title or chunk.chunk_id
        code, note = applicability(editions[key], period, _is_law(chunk), today)
        items.append(Evidence(
            index=len(items) + 1, chunk=chunk, text=text, label=_label(chunk),
            kind=_kind(chunk), applicability=code, applicability_note=note,
            injection_removed=removed,
        ))
    return PreparedEvidence(items=items, period=period, excluded=excluded, notes=notes,
                            premise_hints=_premise_hints(request.question, items))
