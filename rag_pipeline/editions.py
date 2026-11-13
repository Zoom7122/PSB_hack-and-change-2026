"""Даты редакций и выбор применимого контекста без подмены датой выгрузки."""

import calendar
import re
from datetime import date, datetime

MONTHS = (
    "январ", "феврал", "март", "апрел", "ма[йея]", "июн",
    "июл", "август", "сентябр", "октябр", "ноябр", "декабр",
)
DATE = r"(\d{2}\.\d{2}\.\d{4})"


def iso_date(value: str | None) -> str | None:
    """Преобразует календарную дату без догадок о дне или месяце."""
    for pattern in ("%Y-%m-%d", "%d.%m.%Y"):
        try:
            return datetime.strptime(value or "", pattern).date().isoformat()
        except ValueError:
            continue
    return None


def document_period(text: str, meta) -> dict:
    """Извлекает только явно названные сроки действия самой редакции."""
    header = "" if meta.kind == "закон" else re.split(r"\n\s*1[.)]\s", text, maxsplit=1)[0]
    start = re.search(r"(?:действует|вступает в силу)\s+с\s+" + DATE, header, re.I)
    end = re.search(r"^\s*РЕДАКЦИЯ\s+УТРАТИЛА\s+СИЛУ\s+с\s+" + DATE, text, re.I)
    signed = meta.kind == "доп.соглашение" and re.search(r"вступает в силу с даты подписания", text, re.I)
    return {
        "edition_date": iso_date(meta.edition),
        "effective_from": iso_date(start.group(1)) if start else iso_date(meta.date) if signed else None,
        "effective_to": iso_date(end.group(1)) if end else None,
        "validity_evidence": [match.group(0) for match in (start, end) if match]
                             + ([f"{signed.group(0)}; дата документа {meta.date}"] if signed else []),
    }


def question_period(question: str, event_date: str | None = None) -> tuple[str, str] | None:
    """Сохраняет точность даты события: день, месяц или явно указанный год."""
    periods = set()
    for value in re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}\.\d{2}\.\d{4}\b", question):
        if iso_date(value):
            periods.add((iso_date(value), iso_date(value)))
    for index, month in enumerate(MONTHS, start=1):
        for year in re.findall(r"\b" + month + r"[а-я]*\s+(20\d{2})\b", question, re.I):
            last = calendar.monthrange(int(year), index)[1]
            periods.add((date(int(year), index, 1).isoformat(), date(int(year), index, last).isoformat()))
    if not periods:
        for year in re.findall(r"\b(?:в|за|на)\s+(20\d{2})\s*(?:году|год|г\.?)\b", question, re.I):
            periods.add((f"{year}-01-01", f"{year}-12-31"))
    if len(periods) == 1:
        return next(iter(periods))
    if not periods and event_date and iso_date(event_date):
        value = iso_date(event_date)
        return value, value
    return None


def applicability(source: dict, period: tuple[str, str] | None) -> str:
    """Отличает явное действие, выбор датированной редакции и неизвестный период."""
    if period:
        first, last = period
        start, end = source.get("effective_from"), source.get("effective_to")
        if (start and start > last) or (end and end <= first):
            return "outside_event_period"
        if start and start <= first and (not end or last < end):
            return "explicitly_applicable"
        edition = source.get("edition_date")
        if edition and edition <= first and end and last < end:
            return "dated_edition_before_repeal"
        return "unknown"
    return "historical" if source.get("status") == "утратил силу" else "current_or_unknown"


def edition_weight(source: dict, period: tuple[str, str] | None) -> float:
    """Понижает чужую редакцию, сохраняя её для сравнений и объяснения изменений."""
    status = applicability(source, period)
    return {"outside_event_period": 0.15, "historical": 0.45,
            "explicitly_applicable": 1.2, "dated_edition_before_repeal": 1.15}.get(status, 1.0)
