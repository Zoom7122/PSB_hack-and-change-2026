"""Стыковка с соседними модулями: сборка GenerationRequest и выдача источников.

Пайплайн (модуль 1 или solution.py) делает так:

    qp      = await query_processing.process_query(...)          # модуль 2
    ...                                                           # модули 3–4
    request = build_generation_request(rerank_result, qp, request_id=...)
    result  = await answer_service.generate(request)              # модуль 5
    sources = sources_for_response(result, rerank_result)         # для ChatResponse
"""

from __future__ import annotations

from typing import Any


def build_generation_request(rerank_result: dict[str, Any],
                             query_result: dict[str, Any] | None = None, *,
                             question: str | None = None,
                             request_id: str | None = None) -> dict[str, Any]:
    """RerankResult (модуль 4) + QueryProcessingResult (модуль 2) → GenerationRequest.

    Вопрос — `standalone_query` модуля 2 (контракт: в генерацию идёт
    самостоятельный вопрос с учётом истории); дата события — `filters.event_date`.
    Явный `question` перекрывает standalone_query.
    """
    search = (query_result or {}).get("search_request") or {}
    filters = search.get("filters") or {}
    text = question or search.get("standalone_query") or rerank_result.get("question")
    if not text:
        raise ValueError("question is required: pass question= or a ready QueryProcessingResult")
    return {
        "request_id": request_id or rerank_result.get("request_id")
        or (query_result or {}).get("request_id") or "req-local",
        "corpus_version": rerank_result.get("corpus_version") or "",
        "question": text,
        "event_date": filters.get("event_date"),
        "chunks": list(rerank_result.get("chunks") or []),
    }


SOURCE_FIELDS = ("chunk_id", "document_id", "document_title", "revision_id",
                 "section", "page", "source_path")


def sources_for_response(result: dict[str, Any], chunks: list[dict[str, Any]] | dict[str, Any]
                         ) -> list[dict[str, Any]]:
    """SourceReference[] для ChatResponse модуля 1 — только из доверенных чанков.

    Контракт: «API не доверяет реквизитам источников от LLM, заполняет sources
    только из выбранных ранее чанков». Берём chunk_id из citations и
    реквизиты — из исходного списка чанков, в порядке цитирования.
    Дополнительно отдаём `text` и `cite_label`: Frontend показывает фрагмент.
    """
    if isinstance(chunks, dict):
        chunks = chunks.get("chunks") or []
    by_id = {chunk["chunk_id"]: chunk for chunk in chunks if "chunk_id" in chunk}
    sources = []
    for link in result.get("citations", []):
        chunk = by_id.get(link["chunk_id"])
        if chunk is None:
            continue
        source = {name: chunk.get(name) for name in SOURCE_FIELDS}
        source["document_title"] = source["document_title"] or chunk.get("cite_label") or chunk.get("document_id")
        # В SourceReference модуля 1 эти поля — непустые строки.
        source["document_id"] = source["document_id"] or ""
        source["source_path"] = source["source_path"] or ""
        source["text"] = chunk.get("text", "")
        source["cite_label"] = chunk.get("cite_label")
        source["claim_ids"] = link.get("claim_ids", [])
        sources.append(source)
    return sources
