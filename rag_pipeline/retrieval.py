"""Адаптер между Query Processing и поисковым интерфейсом команды."""

from __future__ import annotations

from itertools import zip_longest

from .corpus import Corpus
from .indexing import TeamSearch
from data_processing import config, contract_adapter


def query_variants(question: str, request: dict) -> list[str]:
    """Собирает исходную и подготовленные формулировки без повторов."""
    candidates = [question, request.get("standalone_query", ""), *request.get("search_queries", [])]
    result, seen = [], set()
    for value in candidates:
        text = value.strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _filters(filters: dict) -> dict:
    """Переводит известные типы документов в формат адаптера команды."""
    result = dict(filters or {})
    kind = result.get("document_type")
    if kind:
        result["document_type"] = contract_adapter.DOCUMENT_TYPES.get(kind.lower(), kind.lower())
    return result


def _candidate(corpus: Corpus, hit, query: str) -> dict:
    """Добавляет к проверенному источнику происхождение поискового результата."""
    source = dict(corpus.sources[hit.chunk.chunk_id])
    if any(not source.get(key) for key in ("chunk_id", "document_title", "section", "text", "citation")):
        raise ValueError("Поисковый модуль вернул фрагмент без обязательных метаданных")
    source["matched_queries"] = [query]
    source["matched_by"] = contract_adapter._matched_by(hit)
    source["rrf_score"] = hit.rrf_score
    source["rerank_score"] = hit.rerank_score
    return source


def _matches(corpus: Corpus, hit, filters: dict) -> bool:
    """Сохраняет найденное командой допсоглашение к отфильтрованному договору."""
    item = contract_adapter.chunk_to_candidate(hit, 1)
    if contract_adapter.matches_filters(item, filters):
        return True
    parent = corpus.metas[hit.chunk.doc_id].parent_contract
    requested = filters.get("document_number")
    if not parent or not requested:
        return False
    if contract_adapter.normalize_number(parent) != contract_adapter.normalize_number(requested):
        return False
    related_filters = dict(filters, document_number=None)
    if related_filters.get("document_type") == "contract" and item["document_type"] == "supplement":
        related_filters["document_type"] = None
    return contract_adapter.matches_filters(item, related_filters)


def retrieve(corpus: Corpus, team: TeamSearch, question: str, request: dict,
             limit: int = 6) -> list[dict]:
    """Вызывает поиск команды для формулировок и объединяет его результаты."""
    filters = _filters(request.get("filters") or {})
    rankings = []
    for query in query_variants(question, request):
        hits = team.searcher.search(query, final_k=config.RERANK_CANDIDATES)
        rankings.append([_candidate(corpus, hit, query) for hit in hits
                         if _matches(corpus, hit, filters)])
    results: dict[str, dict] = {}
    for ranked in zip_longest(*rankings):
        for source in ranked:
            if source is None:
                continue
            existing = results.get(source["chunk_id"])
            if existing is None:
                results[source["chunk_id"]] = source
            elif source["matched_queries"][0] not in existing["matched_queries"]:
                existing["matched_queries"].extend(source["matched_queries"])
    return list(results.values())[:limit]
