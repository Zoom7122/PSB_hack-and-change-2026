"""Терпимый разбор JSON-ответов модели.

Модели регулярно оборачивают JSON в ```json, добавляют фразу перед ним,
пишут номера источников строками ("[2]") или называют поля по-русски.
Здесь всё это приводится к одной структуре; неразборчивое — ParseError.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


class ParseError(ValueError):
    """Ответ модели не удалось привести к ожидаемой структуре."""


RE_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)
RE_MARKER = re.compile(r"\s*\[(?:\s*(?:фрагмент\s*)?\d{1,2}\s*[,;–-]?\s*)+\]")
RE_INT = re.compile(r"\d{1,3}")


def extract_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ParseError("empty response")
    candidates = [m.group(1) for m in RE_FENCE.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start, end = candidate.find("{"), candidate.rfind("}")
        if 0 <= start < end:
            try:
                value = json.loads(candidate[start:end + 1])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                continue
    raise ParseError("no JSON object found")


def coerce_sources(raw: Any) -> list[int]:
    """[1, "2", "[3]", "фрагмент 4", 5.0] → [1, 2, 3, 4, 5]; порядок сохраняется."""
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    result: list[int] = []
    for item in items:
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            numbers = [int(item)]
        else:
            numbers = [int(n) for n in RE_INT.findall(str(item))]
        for number in numbers:
            if number not in result:
                result.append(number)
    return result


def clean_sentence(text: str) -> str:
    text = RE_MARKER.sub("", str(text))
    text = re.sub(r"[*_#`]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # «рублей [1], [2].» → «рублей , .» → «рублей.»
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[,;]+(?=[.!?]|$)", "", text)
    return text


_KINDS = {
    "fact": "fact", "факт": "fact", "claim": "fact", "statement": "fact",
    "conclusion": "conclusion", "вывод": "conclusion", "inference": "conclusion",
    "gap": "gap", "missing": "gap", "пробел": "gap", "нет данных": "gap", "limitation": "gap",
}


@dataclass
class DraftSentence:
    text: str
    kind: str
    sources: list[int] = field(default_factory=list)


@dataclass
class Draft:
    status: str
    sentences: list[DraftSentence]
    missing_information: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


RE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«(])")


def parse_draft(text: str) -> Draft:
    data = extract_json(text)
    sentences: list[DraftSentence] = []
    raw_sentences = data.get("sentences") or data.get("предложения")
    if isinstance(raw_sentences, list):
        for item in raw_sentences:
            if isinstance(item, str):
                item = {"text": item}
            if not isinstance(item, dict):
                continue
            body = clean_sentence(item.get("text") or item.get("текст") or "")
            if not body:
                continue
            kind = _KINDS.get(str(item.get("kind") or item.get("type") or "fact").strip().lower(), "fact")
            sentences.append(DraftSentence(
                text=body, kind=kind,
                sources=coerce_sources(item.get("sources", item.get("источники"))),
            ))
    elif isinstance(data.get("answer"), str):
        # Модель проигнорировала схему и вернула текст. Делим на предложения;
        # источники при LLM-проверке может восстановить проверяющий.
        for chunk in RE_SPLIT.split(data["answer"]):
            body = clean_sentence(chunk)
            if body:
                sentences.append(DraftSentence(text=body, kind="fact"))
    else:
        raise ParseError("no sentences in response")

    status = str(data.get("status") or "").strip().lower()
    if status not in {"answered", "insufficient"}:
        has_facts = any(s.kind != "gap" for s in sentences)
        status = "answered" if has_facts else "insufficient"
    missing = data.get("missing_information")
    missing = clean_sentence(missing) if isinstance(missing, str) and missing.strip() else None
    if missing and missing.lower() in {"null", "none", "нет"}:
        missing = None
    if not sentences and not missing:
        raise ParseError("empty draft")
    return Draft(status=status, sentences=sentences, missing_information=missing, raw=data)


@dataclass
class VerifierItem:
    verdict: str
    sources: list[int]
    reason: str


_VERDICTS = {"supported", "unsupported", "gap_ok", "gap_wrong"}


def parse_verification(text: str) -> dict[str, VerifierItem]:
    data = extract_json(text)
    results = data.get("results")
    if not isinstance(results, list):
        raise ParseError("no results in verification")
    parsed: dict[str, VerifierItem] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        verdict = str(item.get("verdict") or "").strip().lower()
        if not sid or verdict not in _VERDICTS:
            continue
        parsed[sid] = VerifierItem(
            verdict=verdict,
            sources=coerce_sources(item.get("sources")),
            reason=clean_sentence(item.get("reason") or "")[:200],
        )
    return parsed
