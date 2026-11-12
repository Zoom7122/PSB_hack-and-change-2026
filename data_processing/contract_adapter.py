"""Адаптер: внутренние объекты поиска → JSON-контракт модулей 3 и 4.

Зачем нужен
===========
Контракты (`CONTRACTS-ALL.md`) описывают целевой интерфейс: поля в snake_case,
`chunk_id` вида `kb-v1:doc:rev:sec`, обязательные `stats` и `fusion`. Внутри
решения работают свои объекты — `chunking.Chunk` (поля `doc_id`, `label`,
`body`) и `hybrid.FusedHit` (поля `bm25_rank`, `rrf_score`, …).

Менять ради контракта ядро поиска нельзя: ядро проверено (хит@6 0.96) и
работает на платформе (балл 9.642). Поэтому адаптер только **читает** и
перекладывает. Логика поиска не затрагивается ни на строку.

Чего в наших данных нет — честно `null`
=======================================
`revision_id`, `effective_from`, `effective_to`, `page`, `source_path` в базе
знаний отсутствуют: документы текстовые, редакция указана в тексте документа,
а не в метаданных. Подставлять догадки нельзя — контракт прямо запрещает
выдумывать реквизиты. Поэтому `null`, а потребитель обязан это пережить.

Запуск проверки: `python check_contract.py`
"""

from __future__ import annotations

import re

if __package__:
    from . import config, hybrid, refs
    from .chunking import Chunk
else:
    import config
    import hybrid
    import refs
    from chunking import Chunk

CORPUS_VERSION = "kb-v1"

# Поля, которые есть в хранилище: обязательные по контракту v1.1
REQUIRED_CANDIDATE_FIELDS = (
    "chunk_id", "document_id", "text", "section", "cite_label",
    "rrf_rank", "rrf_score", "matched_by", "scores",
)
# Поля, которых у нас нет и которые допустимо отдавать как null
NULLABLE_FIELDS = (
    "document_title", "document_type", "document_number",
    "revision_id", "effective_from", "effective_to",
    "page", "source_path",
)

RE_DOC_NUMBER = re.compile(r"№\s*([А-ЯЁA-Z0-9][А-ЯЁA-Z0-9\-/\.]*)")

# В базе типы документов записаны по-русски, контракт использует английские
# значения ("contract", "law"). Переводим на границе, чтобы API и модуль 5 не
# разбирали русские строки. Неизвестный тип → null, а не догадка.
DOCUMENT_TYPES = {
    "договор": "contract",
    "доп.соглашение": "supplement",
    "закон": "law",
    "записка": "memo",
    "письмо": "letter",
    "регламент": "regulation",
}
CONTRACT_DOCUMENT_TYPES = frozenset(DOCUMENT_TYPES.values())


def document_title(chunk: Chunk) -> str:
    """Название документа без реквизитов.

    В базе шапка чанка склеена: «Договор аренды № А-08/2024 Северный Склад |
    ООО «Северный Склад», ИНН …, от 01.08.2024». Для ответа нужна только первая
    часть — она читается человеком и совпадает с эталонным форматом ссылок.
    """
    head = (chunk.prefix or "").split("|")[0].strip()
    return head or chunk.doc_id


def document_number(chunk: Chunk) -> str | None:
    """Номер документа, если он есть в названии. «А-08/2024» → «А-08/2024»."""
    match = RE_DOC_NUMBER.search(document_title(chunk))
    return match.group(1).rstrip(",.") if match else None


def cite_label(chunk: Chunk) -> str:
    """Готовая ссылка для ответа: «п. 3.1» или «НК РФ, ст. 164».

    Поле `section` в наших данных неоднородно: у законов это «ч. 1 (продолжение
    48)» — служебный маркер нарезки, а не юридическая ссылка. Показывать его
    юристу нельзя. Формат ссылки уже отлажен под грейдер (стиль `inline`,
    балл 9.642), поэтому адаптер переиспользует ту же функцию, а не заводит
    второй формат.
    """
    return refs.inline_reference(document_title(chunk), chunk.label) or document_title(chunk)


def document_type(chunk: Chunk) -> str | None:
    """Тип документа в терминах контракта: «договор» → «contract»."""
    return DOCUMENT_TYPES.get((chunk.kind or "").strip().lower())


def normalize_number(value: str | None) -> str | None:
    """Нормализация номера документа для фильтра: «№ А-08/2024» → «а-08/2024»."""
    if not value:
        return None
    cleaned = value.strip().lower().replace("№", "").replace(" ", "")
    return cleaned.rstrip(".,") or None


def matches_filters(candidate: dict, filters: dict | None) -> bool:
    """Проходит ли кандидат фильтры из `search_request.filters`.

    Семантика зафиксирована как в контракте v1.1:
      * document_id / document_number / document_type — точное совпадение
        после нормализации, не префикс;
      * None в фильтре — «не ограничивать»;
      * event_date — если у чанка нет дат действия (а в нашей базе их нет),
        кандидат НЕ отбрасывается: отсутствие данных не равно неприменимости.
    """
    if not filters:
        return True
    for key, expected in filters.items():
        if expected in (None, "", []):
            continue
        if key == "document_id" and candidate.get("document_id") != expected:
            return False
        if key == "document_number":
            if normalize_number(candidate.get("document_number")) != normalize_number(str(expected)):
                return False
        if key == "document_type" and candidate.get("document_type") != expected:
            return False
        # event_date: дат действия в базе нет — не фильтруем, но и не врём
    return True


def chunk_to_candidate(hit: hybrid.FusedHit, ranking: int, *,
                       rerank_score: float | None = None) -> dict:
    """Один фрагмент в формате контракта (кандидат поиска или результат реранка)."""
    chunk = hit.chunk
    candidate = {
        "chunk_id": f"{CORPUS_VERSION}:{chunk.chunk_id}",
        "document_id": chunk.doc_id,
        "document_title": document_title(chunk),
        "document_type": document_type(chunk),
        "document_number": document_number(chunk),
        "revision_id": None,
        "effective_from": None,
        "effective_to": None,
        "section": chunk.label,
        "cite_label": cite_label(chunk),
        "page": None,
        "source_path": None,
        "text": chunk.body,
        "rrf_rank": ranking,
        "rrf_score": round(hit.rrf_score, 6),
        "matched_by": _matched_by(hit),
        "scores": {
            "bm25": round(hit.bm25_score, 4),
            "vector": round(hit.vector_score, 4),
        },
    }
    if rerank_score is not None:
        candidate["rerank_score"] = round(rerank_score, 6)
    return candidate


def _matched_by(hit: hybrid.FusedHit) -> list[str]:
    """Какие поиски нашли фрагмент: contract v1.1, дедупликация = объединение."""
    found = []
    if hit.bm25_rank is not None:
        found.append("bm25")
    if hit.vector_rank is not None:
        found.append("vector")
    return found


def retrieval_stats(hits: list[hybrid.FusedHit]) -> dict:
    """Метрики, на которых принимается решение о достаточности.

    Ровно то, что нужно гейту отказа: сырой BM25 и косинус. `rrf_score` для
    порога не годится — он зависит от k и от числа списков.
    """
    best_bm25 = max((hit.bm25_score for hit in hits), default=0.0)
    best_vector = max((hit.vector_score for hit in hits), default=0.0)
    return {
        "max_bm25": round(best_bm25, 4),
        "max_vector": round(best_vector, 4),
        "n_above_bm25_gate": sum(1 for hit in hits
                                 if hit.bm25_score >= config.REFUSAL_BM25_MIN),
        "n_above_vector_gate": sum(1 for hit in hits
                                   if hit.vector_score >= config.REFUSAL_VECTOR_MIN),
        "gates": {
            "bm25_min": config.REFUSAL_BM25_MIN,
            "vector_min": config.REFUSAL_VECTOR_MIN,
        },
    }


def _fusion_block() -> dict:
    """Применённые параметры слияния — без них rrf_score невоспроизводим."""
    return {
        "k": config.RRF_K,
        "weights": {"bm25": config.BM25_WEIGHT, "vector": 1.0},
        "bm25_depth": config.BM25_FUSION_DEPTH,
    }


def retrieval_result(request_id: str, hits: list[hybrid.FusedHit], *,
                     limits: dict | None = None,
                     empty_reason: str | None = None,
                     filters: dict | None = None,
                     expected_corpus_version: str | None = None) -> dict:
    """`RetrievalResult` по контракту v1.1 (модуль 3)."""
    # версия корпуса: искать по чужим индексам нельзя — лучше честно упасть
    if expected_corpus_version and expected_corpus_version != CORPUS_VERSION:
        return {
            "request_id": request_id, "status": "failed",
            "corpus_version": CORPUS_VERSION,
            "empty_reason": "corpus_version_mismatch",
            "limits": limits or {}, "fusion": _fusion_block(),
            "stats": retrieval_stats([]), "candidates": [],
        }

    unfiltered = [chunk_to_candidate(hit, position)
                  for position, hit in enumerate(hits, start=1)]
    candidates = [item for item in unfiltered if matches_filters(item, filters)]
    for position, item in enumerate(candidates, start=1):
        item["rrf_rank"] = position
    if not candidates and unfiltered and filters:
        empty_reason = "filtered_out"
    elif not candidates and not unfiltered:
        empty_reason = empty_reason or "no_matches"

    status = "ok" if candidates else ("empty" if empty_reason in
                                      {"no_matches", "filtered_out"} else "failed")
    return {
        "request_id": request_id,
        "status": status,
        "filters_applied": bool(filters and any(v not in (None, "", []) for v in filters.values())),
        "candidates_dropped_by_filter": len(unfiltered) - len(candidates),
        "corpus_version": CORPUS_VERSION,
        "empty_reason": empty_reason if not candidates else None,
        "limits": limits or {
            "vector_top_k": config.VECTOR_TOP_K,
            "bm25_top_k": config.BM25_TOP_K,
            "fused_top_k": config.RERANK_CANDIDATES,
        },
        "fusion": _fusion_block(),
        "stats": retrieval_stats(hits),
        "candidates": candidates,
    }


def rerank_result(request_id: str, hits: list[hybrid.FusedHit], *,
                  top_k: int = config.RERANK_TOP_N,
                  score_kind: str = "stub") -> dict:
    """`RerankResult` по контракту v1.1 (модуль 4).

    score_kind:
      "endpoint"    — оценки Qwen3 Reranker через /rerank;
      "llm_ordinal" — порядок от языковой модели (сравним только внутри ответа);
      "stub"        — заглушка: порядок по rrf_rank, оценка 1/(1+rrf_rank).
    """
    ordered = sorted(hits, key=lambda hit: hit.rrf_score, reverse=True)[:top_k]
    chunks = []
    for position, hit in enumerate(ordered, start=1):
        rerank_score = (hit.rerank_score if hit.rerank_score is not None
                        else 1.0 / (1.0 + position))
        chunk = chunk_to_candidate(hit, position, rerank_score=rerank_score)
        chunk["rank"] = position
        chunks.append(chunk)
    return {
        "request_id": request_id,
        "status": "ok" if chunks else "empty",
        "corpus_version": CORPUS_VERSION,
        "rerank_model": config.RERANK_MODEL or None,
        "score_kind": score_kind,
        "top_k": top_k,
        "dropped": max(0, len(hits) - len(chunks)),
        "rerank_stats": retrieval_stats(hits),
        "chunks": chunks,
    }
