"""Компактный и очищенный контекст модели со ссылками на исходный корпус."""

from data_processing.textproc import evidence_text
from .models import Claim


def model_sources(sources: list[dict]) -> list[dict]:
    """Заменяет длинные ID локальными псевдонимами и исключает поисковой шум."""
    result = []
    keys = ("document_title", "document_date", "section", "edition", "edition_date", "effective_from",
            "effective_to", "validity_evidence", "applicability", "valid_as_of", "parent_contract")
    for index, source in enumerate(sources, start=1):
        item = {key: source[key] for key in keys if source.get(key) is not None}
        item.update(chunk_id=f"S{index}", text=evidence_text(source.get("body", source["text"])))
        result.append(item)
    return result


def restore_claim_ids(claims: list[Claim], sources: list[dict]) -> list[Claim]:
    """Разрешает только переданные псевдонимы в стабильные ID фрагментов."""
    aliases = {f"S{index}": source["chunk_id"] for index, source in enumerate(sources, start=1)}
    return [Claim(text=claim.text, chunk_ids=list(dict.fromkeys(
        aliases.get(chunk_id, chunk_id) for chunk_id in claim.chunk_ids
    ))) for claim in claims]


def previous_draft(claims: list[Claim], sources: list[dict]) -> list[dict]:
    """Возвращает исправляемые утверждения с теми же псевдонимами источников."""
    aliases = {source["chunk_id"]: f"S{index}" for index, source in enumerate(sources, start=1)}
    return [{"text": claim.text, "chunk_ids": [aliases.get(cid, cid) for cid in claim.chunk_ids]}
            for claim in claims]
