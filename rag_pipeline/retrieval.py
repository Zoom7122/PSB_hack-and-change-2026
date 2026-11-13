"""Общее ранжирование формулировок и подготовка подтверждающего контекста."""

import os
from copy import copy

from data_processing import config, contract_adapter, textproc
from .context import document_anchors, expand_context
from .editions import question_period, edition_weight
from data_processing.hybrid import FusedHit


def query_variants(question: str, request: dict) -> list[str]:
    """Собирает формулировки, не подмешивая старые ответы ассистента в поиск."""
    raw = [] if "\n" in question and "ссистент:" in question else [question]
    candidates = [request.get("standalone_query", ""), *request.get("search_queries", []), *raw]
    result, seen = [], set()
    for value in candidates:
        text = value.strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _filters(filters: dict) -> dict:
    """Переводит известные типы; неизвестный тип не обнуляет весь поиск."""
    result = dict(filters or {})
    kind = (result.get("document_type") or "").lower()
    result["document_type"] = contract_adapter.DOCUMENT_TYPES.get(kind, kind) or None
    if result["document_type"] not in contract_adapter.CONTRACT_DOCUMENT_TYPES:
        result["document_type"] = None
    return result


def _matches(corpus, hit, filters: dict) -> bool:
    """Применяет точный фильтр, сохраняя допсоглашение к основному договору."""
    item = contract_adapter.chunk_to_candidate(hit, 1)
    if contract_adapter.matches_filters(item, filters):
        return True
    parent = corpus.metas[hit.chunk.doc_id].parent_contract
    requested = filters.get("document_number")
    if not parent or not requested or (
        contract_adapter.normalize_number(parent) != contract_adapter.normalize_number(requested)
    ):
        return False
    related = dict(filters, document_number=None)
    if related.get("document_type") == "contract" and item["document_type"] == "supplement":
        related["document_type"] = None
    return contract_adapter.matches_filters(item, related)


def _document_rankings(corpus, searcher, queries: list[str]) -> list[tuple[str, list]]:
    """Добавляет положения из релевантных полных документов к поиску чанков."""
    if not hasattr(searcher, "lexical"):
        return []
    rankings = []
    for query in queries[:2]:
        focus = _content_query(query) or query
        documents = corpus.document_index.search(focus, k=3)
        hits = corpus.content_index.search(focus, k=len(corpus.chunks))
        rankings.append((query, [FusedHit(chunk=corpus.chunks[hit.position], rrf_score=1 / (30 + hit.rank),
                                        bm25_rank=hit.rank, bm25_score=hit.score) for hit in hits[:20]]))
        ranked = []
        for document in documents:
            ranked.extend(FusedHit(chunk=corpus.chunks[hit.position], rrf_score=1 / (30 + hit.rank),
                                   bm25_rank=hit.rank, bm25_score=hit.score)
                          for hit in [item for item in hits if item.chunk.doc_id == document.chunk.doc_id][:4])
        rankings.append((query, ranked))
    return rankings


def _content_query(query: str) -> str:
    """Убирает суммы и календарные реквизиты только из дополнительной формулировки."""
    generic = {"договор", "млн", "рубл", "год", "нужн", "заключал", "наш", "нов"}
    months = ("январ", "феврал", "март", "апрел", "май", "июн", "июл", "август",
              "сентябр", "октябр", "ноябр", "декабр")
    return " ".join(token for token in textproc.tokenize(query) if not any(char.isdigit() for char in token)
                    and token not in generic and not token.startswith(months))


def _combine(corpus, rankings: list[list], queries: list[str], filters: dict,
             question: str, period, primary_count: int) -> tuple[list, dict]:
    """Складывает ранги по ID, учитывая редакцию и явно названный документ."""
    scores, hits, provenance = {}, {}, {}
    anchors = document_anchors(corpus, question)
    for index, ranking in enumerate(rankings):
        weight = 1.3 if index == 0 else (0.6 if index >= primary_count else 1.0)
        query = queries[index]
        for rank, hit in enumerate(ranking, start=1):
            if not _matches(corpus, hit, filters):
                continue
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + weight / (30 + rank)
            if cid not in hits:
                hits[cid] = copy(hit)
            else:
                for field in ("bm25_rank", "vector_rank"):
                    ranks = [value for value in (getattr(hits[cid], field), getattr(hit, field)) if value is not None]
                    setattr(hits[cid], field, min(ranks) if ranks else None)
                hits[cid].bm25_score = max(hits[cid].bm25_score, hit.bm25_score)
                hits[cid].vector_score = max(hits[cid].vector_score, hit.vector_score)
            provenance.setdefault(cid, []).append(query)
    for cid in scores:
        source = corpus.sources[cid]
        scores[cid] *= edition_weight(source, period)
        if anchors:
            related = any(corpus.metas[key].number == source.get("parent_contract") for key in anchors)
            scores[cid] *= 1.8 if source["document_id"] in anchors or related else 0.5
        hits[cid].rrf_score = scores[cid]
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [hits[cid] for cid in ranked], provenance


def _candidate(corpus, hit, queries: list[str]) -> dict:
    """Сохраняет текст корпуса, ID и фактическое происхождение результата."""
    source = dict(corpus.sources[hit.chunk.chunk_id])
    if any(not source.get(key) for key in ("chunk_id", "document_title", "section", "text", "citation")):
        raise ValueError("Поисковый модуль вернул фрагмент без обязательных метаданных")
    source.update(matched_queries=list(dict.fromkeys(queries)),
                  matched_by=contract_adapter._matched_by(hit), rrf_score=hit.rrf_score,
                  rerank_score=hit.rerank_score, bm25_score=hit.bm25_score, vector_score=hit.vector_score)
    return source


def retrieve(corpus, team, question: str, request: dict, limit: int | None = None) -> list[dict]:
    """Объединяет поиск и один реранж, затем добавляет связанные положения."""
    filters = _filters(request.get("filters") or {})
    queries = query_variants(question, request)
    if not queries:
        return []
    searcher = team.searcher
    depth = max(30, config.RERANK_CANDIDATES)
    if hasattr(searcher, "search_many"):
        rankings = searcher.search_many(queries, final_k=depth)
    else:
        rankings = [searcher.search(query, final_k=depth) for query in queries]
    extras = _document_rankings(corpus, searcher, queries)
    rankings.extend(ranking for _, ranking in extras)
    origins = [*queries, *(query for query, _ in extras)]
    period = question_period(question, filters.get("event_date"))
    hits, provenance = _combine(corpus, rankings, origins, filters, question, period, len(queries))
    if hasattr(searcher, "rerank_candidates"):
        hits = searcher.rerank_candidates(request.get("standalone_query") or question, hits)
    seeds = [_candidate(corpus, hit, provenance[hit.chunk.chunk_id]) for hit in hits[:limit or 8]]
    if limit is not None:
        return seeds
    return expand_context(corpus, seeds, question, period,
                          max_sources=int(os.getenv("CONTEXT_MAX_SOURCES", "24")),
                          max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "24000")))
