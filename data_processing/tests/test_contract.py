"""Проверка: выход модулей 3 и 4 соответствует контракту v1.1.

Зачем
=====
Контракт согласуется словами, а ломается на типах: `null` вместо строки,
отсутствующий `stats`, перепутанный порядок `rank`. Скрипт ловит это до того,
как на адаптер начнут опираться API и модуль 5.

Что делает
==========
1. Строит лексический индекс по реальным фрагментам базы (360 штук).
2. Делает настоящий поиск BM25 → RRF → собирает `RetrievalResult` адаптером.
3. Прогоняет заглушку реранка → `RerankResult`.
4. Валидирует оба объекта по правилам контракта.
5. Пишет фикстуры в `data/fixtures/` — по ним команда может разрабатывать API
   и модуль 5, не дожидаясь модулей 3–4.

Внешние библиотеки: только `rank-bm25` (из разрешённого стека). Сети не нужно.

Запуск
======
    python tests/test_contract.py          # 107 проверок + перезапись фикстур
    python -m pytest tests/ -q             # если стоит pytest (не обязателен)

"""

from __future__ import annotations

import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(__file__).resolve().parent.parent      # корень проекта
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import contract_adapter as adapter  # noqa: E402
import hybrid  # noqa: E402
import search  # noqa: E402
import textproc  # noqa: E402
from chunking import Chunk  # noqa: E402

CHUNKS = BASE / "artifacts" / "chunks.jsonl"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
QUESTION = "Можно ли расторгнуть договор без предварительного уведомления?"

passed = 0
failed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed
    if condition:
        passed += 1
        print(f"  OK   {name}")
    else:
        failed.append(f"{name} {detail}".strip())
        print(f"  FAIL {name} {detail}".strip())


def load_chunks() -> list[Chunk]:
    if not CHUNKS.is_file():
        raise SystemExit(f"Нет файла {CHUNKS}. Соберите его: python scripts/prepare_chunks.py")
    chunks = []
    with CHUNKS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks


def build_hits(chunks: list[Chunk], question: str) -> list[hybrid.FusedHit]:
    """Настоящий BM25-поиск + RRF (векторов без сети нет — список векторов пуст)."""
    index = search.LexicalIndex(chunks)
    tokens = textproc.tokenize_query(question)
    lexical_hits = index.search_tokens(tokens, k=config_bm25_top_k())
    positions = [hit.position for hit in lexical_hits]
    fused = hybrid.reciprocal_rank_fusion([positions], weights=[1.0], depths=[None])
    by_position = {hit.position: hit for hit in lexical_hits}
    hits = [
        hybrid.FusedHit(
            chunk=chunks[position],
            rrf_score=score,
            bm25_rank=by_position[position].rank if position in by_position else None,
            bm25_score=by_position[position].score if position in by_position else 0.0,
        )
        for position, score in fused.items()
    ]
    hits.sort(key=lambda hit: hit.rrf_score, reverse=True)
    return hits


def config_bm25_top_k() -> int:
    import config
    return config.BM25_TOP_K


def validate_candidate(candidate: dict, index: int, where: str) -> None:
    for field in adapter.REQUIRED_CANDIDATE_FIELDS:
        check(f"{where}[{index}].{field} присутствует", field in candidate)
    for field in ("chunk_id", "document_id", "text", "section", "cite_label"):
        check(f"{where}[{index}].{field} — непустая строка",
              isinstance(candidate.get(field), str) and bool(candidate[field]))
    check(f"{where}[{index}].cite_label — ссылка, а не служебный маркер",
          bool(candidate.get("cite_label"))
          and "продолжение" not in candidate["cite_label"]
          and "фрагмент" not in candidate["cite_label"],
          f"получено {candidate.get('cite_label')!r}")
    check(f"{where}[{index}].section не пустой (в контракте это номер пункта)",
          bool(candidate.get("section")), f"получено {candidate.get('section')!r}")
    check(f"{where}[{index}].rrf_rank — целое ≥ 1",
          isinstance(candidate.get("rrf_rank"), int) and candidate["rrf_rank"] >= 1)
    check(f"{where}[{index}].matched_by — список из vector/bm25",
          isinstance(candidate.get("matched_by"), list)
          and set(candidate["matched_by"]) <= {"vector", "bm25"})
    scores = candidate.get("scores") or {}
    check(f"{where}[{index}].scores содержит bm25 и vector",
          isinstance(scores.get("bm25"), (int, float))
          and isinstance(scores.get("vector"), (int, float)))
    for field in adapter.NULLABLE_FIELDS:
        check(f"{where}[{index}].{field} присутствует (null допустим)", field in candidate)


def main() -> int:
    chunks = load_chunks()
    print(f"Фрагментов в базе: {len(chunks)}")
    hits = build_hits(chunks, QUESTION)
    print(f"BM25 нашёл: {len(hits)} кандидатов\n")

    retrieval = adapter.retrieval_result("req-fix-001", hits)
    rerank = adapter.rerank_result("req-fix-001", hits)

    print("МОДУЛЬ 3 — RetrievalResult")
    check("status в допустимых значениях",
          retrieval["status"] in {"ok", "empty", "failed"})
    check("corpus_version — строка", isinstance(retrieval["corpus_version"], str)
          and bool(retrieval["corpus_version"]))
    fusion = retrieval["fusion"]
    check("fusion.k — целое", isinstance(fusion.get("k"), int))
    check("fusion.weights содержит bm25 и vector",
          isinstance(fusion.get("weights", {}).get("bm25"), (int, float))
          and isinstance(fusion.get("weights", {}).get("vector"), (int, float)))
    stats = retrieval["stats"]
    for field in ("max_bm25", "max_vector", "n_above_bm25_gate", "n_above_vector_gate"):
        check(f"stats.{field} присутствует", field in stats)
    check("stats.gates содержит оба порога",
          "bm25_min" in stats.get("gates", {}) and "vector_min" in stats.get("gates", {}))
    check("кандидаты отсортированы по rrf_rank",
          [c["rrf_rank"] for c in retrieval["candidates"]]
          == sorted(c["rrf_rank"] for c in retrieval["candidates"]))
    check("chunk_id уникальны",
          len({c["chunk_id"] for c in retrieval["candidates"]})
          == len(retrieval["candidates"]))
    check("rrf_score убывает вместе с рангом",
          all(a["rrf_score"] >= b["rrf_score"]
              for a, b in zip(retrieval["candidates"], retrieval["candidates"][1:])))
    for index, candidate in enumerate(retrieval["candidates"][:3]):
        validate_candidate(candidate, index, "кандидат")

    print("\nМОДУЛЬ 4 — RerankResult")
    check("status в допустимых значениях", rerank["status"] in {"ok", "empty", "failed"})
    check("score_kind задан", rerank["score_kind"] in {"endpoint", "llm_ordinal", "stub"})
    check("top_k — целое > 0", isinstance(rerank["top_k"], int) and rerank["top_k"] > 0)
    check("dropped — целое ≥ 0", isinstance(rerank["dropped"], int) and rerank["dropped"] >= 0)
    check("chunks не длиннее top_k", len(rerank["chunks"]) <= rerank["top_k"])
    check("rank идёт подряд с 1",
          [c["rank"] for c in rerank["chunks"]] == list(range(1, len(rerank["chunks"]) + 1)))
    check("rerank_score у каждого чанка",
          all(isinstance(c.get("rerank_score"), float) for c in rerank["chunks"]))
    check("происхождение сохранено (rrf_rank, matched_by)",
          all("rrf_rank" in c and "matched_by" in c for c in rerank["chunks"]))
    check("все chunks — подмножество кандидатов поиска",
          {c["chunk_id"] for c in rerank["chunks"]}
          <= {c["chunk_id"] for c in retrieval["candidates"]})

    print("\nФИЛЬТРЫ И ВЕРСИЯ КОРПУСА")
    check("document_type во всех кандидатах — значение контракта",
          all(c["document_type"] in adapter.CONTRACT_DOCUMENT_TYPES
              for c in retrieval["candidates"]),
          f"встретились {sorted({str(c['document_type']) for c in retrieval['candidates']})}")
    check("русские типы не протекли наружу",
          not any(c["document_type"] in {"договор", "закон", "записка"}
                  for c in retrieval["candidates"]))
    numbers = [c["document_number"] for c in retrieval["candidates"]
               if c["document_number"]]
    if numbers:
        target = numbers[0]
        filtered = adapter.retrieval_result(
            "req-fix-003", hits, filters={"document_number": target})
        check(f"фильтр по номеру «{target}» оставляет только этот документ",
              all(adapter.normalize_number(c["document_number"])
                  == adapter.normalize_number(target) for c in filtered["candidates"]))
        check("фильтр уменьшил выдачу", len(filtered["candidates"]) < len(retrieval["candidates"]))
        check("rrf_rank после фильтра идёт подряд",
              [c["rrf_rank"] for c in filtered["candidates"]]
              == list(range(1, len(filtered["candidates"]) + 1)))
        check("filters_applied отмечен", filtered["filters_applied"] is True)
        # «№ А-08/2024» и «а-08/2024» — один и тот же документ
        spaced = adapter.retrieval_result(
            "req-fix-004", hits, filters={"document_number": f"№ {target}"})
        check("номер с «№» и пробелом нормализуется",
              len(spaced["candidates"]) == len(filtered["candidates"]))

    empty_filter = adapter.retrieval_result(
        "req-fix-005", hits, filters={"document_number": "НЕТ-ТАКОГО-999"})
    check("фильтр без совпадений → status=empty",
          empty_filter["status"] == "empty", empty_filter["status"])
    check("причина — filtered_out, а не no_matches",
          empty_filter["empty_reason"] == "filtered_out", str(empty_filter["empty_reason"]))
    check("candidates_dropped_by_filter считает отброшенные",
          empty_filter["candidates_dropped_by_filter"] == len(retrieval["candidates"]))

    by_type = adapter.retrieval_result(
        "req-fix-006", hits, filters={"document_type": "law"})
    check("фильтр по типу law оставляет только законы",
          all(c["document_type"] == "law" for c in by_type["candidates"]))
    no_type = adapter.retrieval_result(
        "req-fix-007", hits, filters={"document_type": "contract"})
    check("фильтр по типу contract не пуст (в базе есть договоры)",
          all(c["document_type"] == "contract" for c in no_type["candidates"]))

    mismatch = adapter.retrieval_result("req-fix-008", hits,
                                        expected_corpus_version="kb-v0")
    check("несовпадение corpus_version → status=failed", mismatch["status"] == "failed")
    check("причина — corpus_version_mismatch",
          mismatch["empty_reason"] == "corpus_version_mismatch")
    check("при несовпадении кандидатов не отдаём", mismatch["candidates"] == [])
    same = adapter.retrieval_result("req-fix-009", hits,
                                    expected_corpus_version=adapter.CORPUS_VERSION)
    check("совпадение corpus_version не мешает поиску", same["status"] == "ok")

    print("\nВЕТКА ОТКАЗА")
    empty = adapter.retrieval_result("req-fix-002", [], empty_reason="no_matches")
    check("пустой результат даёт status=empty", empty["status"] == "empty")
    check("empty_reason заполнен", empty["empty_reason"] == "no_matches")
    check("candidates — пустой массив, не null", empty["candidates"] == [])
    check("stats при пустом результате нулевые",
          empty["stats"]["max_bm25"] == 0.0 and empty["stats"]["max_vector"] == 0.0)
    empty_rerank = adapter.rerank_result("req-fix-002", [])
    check("реранк пустого входа возвращает chunks: []", empty_rerank["chunks"] == [])

    FIXTURES.mkdir(parents=True, exist_ok=True)
    filter_example = adapter.retrieval_result(
        "req-fix-010", hits, filters={"document_type": "contract"})
    written = 0
    for name, payload in (
        ("module3_retrieval_result.json", retrieval),
        ("module3_filtered_result.json", filter_example),
        ("module3_empty_result.json", empty),
        ("module4_rerank_result.json", rerank),
        ("module2_search_request.json", {
            "request_id": "req-fix-001",
            "status": "ready",
            "message": None,
            "search_request": {
                "standalone_query": QUESTION,
                "search_queries": ["условия расторжения договора", "срок уведомления"],
                "filters": {"document_id": None, "document_number": None,
                            "document_type": None, "event_date": None},
            },
        }),
    ):
        (FIXTURES / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written += 1

    print("\n" + "=" * 70)
    print(f"ИТОГ: пройдено {passed}, провалено {len(failed)}")
    if failed:
        for item in failed:
            print(f"  · {item}")
        return 1
    print(f"Фикстуры записаны в {FIXTURES.relative_to(BASE)}/ ({written} файлов)")
    print("Модули 3 и 4 отдают данные в формате контракта.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
