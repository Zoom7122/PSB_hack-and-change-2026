"""Подбор параметров слияния BM25 и векторов — по цифрам, а не на глаз.

Запуск:
    python tune_fusion.py                  # сетка: вес × глубина × RRF_K
    python tune_fusion.py --selftest       # проверить логику выбора без сети
    python tune_fusion.py --vector-k 30    # считать по более узкому списку

Что делает
==========
1. Загружает уже собранный векторный индекс из artifacts/ (если он есть) —
   документы заново НЕ кодируются.
2. Кодирует 17 демо-вопросов (17 запросов к API, копейки).
3. Считает BM25-места локально, без сети.
4. Перебирает вес BM25, глубину и постоянную сглаживания RRF_K.
5. Печатает таблицу и готовые строки для .env.

Три параметра и что каждый делает
=================================
**Вес BM25** (`BM25_WEIGHT`) — чей голос громче. 0 = BM25 выключен совсем,
1.0 = равноправно.

**Глубина** (`BM25_FUSION_DEPTH`) — сколько первых мест BM25 учитывать. В хвосте
у него случайные совпадения по одному общему слову.

**RRF_K** — насколько резко места разделены между собой. Это ключ к hit@1, и его
обычно упускают:

| RRF_K | 1-е место | 2-е место | разница | что происходит |
|---|---:|---:|---:|---|
| 60 (классика) | 1/61 = 0.0164 | 1/62 = 0.0161 | 0.0003 | места почти равны, слабый голос легко переставляет сильного |
| 10 | 1/11 = 0.0909 | 1/12 = 0.0833 | 0.0076 | сильный поиск держит порядок, слабый лишь корректирует |

При K = 60 (значение из классической статьи) разница между соседними местами
составляет 2 %. Значит голос BM25 с весом 0.1 (то есть 10 %) легко переставляет
фрагменты, которые векторы поставили на 1-е и 2-е место. Отсюда и просадка
hit@1 у гибрида при том, что hit@5 идеален.

При K = 10 разница между местами — уже 9 %, и малый вес BM25 работает как
задумано: уточняет, но не переворачивает.

Какой метрикой решать
=====================
Решающая — **hit@6** (ровно столько фрагментов уходит в промпт). Если hit@6
одинаков, ответ не изменится, и выбирать нечего.

Второй по важности — **hit@1**: он влияет на качество ответа через внимание
модели (фрагмент в начале контекста модель использует охотнее). Но это не
вопрос «попал/не попал», поэтому при равном hit@6 предпочитаем вариант с
большим hit@1, а не жертвуем BM25 ради него.

Почему нельзя брать вес 0
=========================
Вес 0 = BM25 выключен: остаётся один векторный поиск. На демо он выглядит лучше
по hit@1, но запросы «пункт 3.1», «ИНН 7701234567», «450 000» держатся именно
на точном совпадении токенов, а векторный поиск в цифрах слаб. BM25 держим
включённым, но с малым весом.

Как не обмануть себя
====================
Вопросов 17: один вопрос = 0.06. Разница меньше 0.05 — шум.
  * не берите значения, которые выиграли только на 1–2 вопросах;
  * если варианты равны — оставьте текущие настройки;
  * и проверьте выбранное на демо-наборе целиком, а не на подмножестве.
"""

from __future__ import annotations

import argparse
import csv
import sys

import chunking
import config
import ingest
import query_prep
import search
import textproc
from hybrid import reciprocal_rank_fusion
from llm_client import LLMError, make_client
from vector_index import build_or_load

# Вопросы демо-набора с фрагментом-эталоном (те же, что в check_step4.py).
from check_step4 import GOLDEN

# Сколько фрагментов уходит в промпт на шаге 6 — по этому числу и решаем.
CONTEXT_K = 6
# Вес ниже этого считаем «BM25 практически выключен».
MIN_USEFUL_WEIGHT = 0.05


def metrics(ranks: list[int | None]) -> dict[str, float]:
    total = len(ranks)
    return {
        "hit@1": sum(1 for r in ranks if r and r <= 1) / total,
        "hit@5": sum(1 for r in ranks if r and r <= 5) / total,
        f"hit@{CONTEXT_K}": sum(1 for r in ranks if r and r <= CONTEXT_K) / total,
        "mrr": sum(1 / r for r in ranks if r) / total,
    }


def rank_of(found: list[str], needle: str) -> int | None:
    for place, body in enumerate(found, start=1):
        if needle in body:
            return place
    return None


def choose(
    rows: list[tuple[float, int | None, int, dict[str, float]]],
    current: tuple[float, int | None, int],
) -> tuple[float, int | None, int, dict[str, float], str]:
    """Выбрать параметры по правилам, а не по максимуму одной цифры.

    rows: (вес, глубина, RRF_K, метрики).

    Правила:
      1. отбрасываем варианты, которые теряют по hit@6 — это «не попало в контекст»;
      2. если текущие настройки в этом множестве — оставляем их: не трогаем
         работающее без причины (защита от подгонки под 17 вопросов);
      3. иначе среди вариантов с рабочим весом BM25 берём лучший по hit@1,
         при равенстве — по hit@5.
    """
    key = f"hit@{CONTEXT_K}"
    best_ctx = max(m[key] for _, _, _, m in rows)
    contenders = [r for r in rows if r[3][key] >= best_ctx - 1e-9]

    for weight, depth, rrf_k, m in contenders:
        if (abs(weight - current[0]) < 1e-9 and (depth or 0) == (current[1] or 0)
                and rrf_k == current[2]):
            return weight, depth, rrf_k, m, (
                f"текущие настройки не хуже остальных по {key} — оставляем как есть")

    useful = [r for r in contenders if r[0] >= MIN_USEFUL_WEIGHT] or contenders
    weight, depth, rrf_k, m = max(useful, key=lambda r: (r[3]["hit@1"], r[3]["hit@5"], -r[0]))
    return weight, depth, rrf_k, m, (
        "BM25 остаётся включённым: на числовых запросах он надёжнее векторов"
        if weight >= MIN_USEFUL_WEIGHT else
        "все варианты с рабочим весом BM25 проигрывают — база и вопросы чисто смысловые")


def selftest() -> int:
    """Проверка логики выбора на выдуманных данных — без сети и без API."""
    ok = True

    def row(w, d, k, h1, h6):
        return (w, d, k, {"hit@1": h1, "hit@5": max(h1, h6), f"hit@{CONTEXT_K}": h6, "mrr": h1})

    # случай с нашей базы: вес 0 лучше по hit@1, но контекст одинаковый
    rows = [row(0.0, 5, 60, 0.88, 1.0), row(0.1, 5, 60, 0.76, 1.0), row(0.3, 5, 60, 0.70, 0.94)]
    w, d, k, m, why = choose(rows, current=(0.1, 5, 60))
    ok &= (w, d, k) == (0.1, 5, 60)
    print(f"  {'OK  ' if (w, d, k) == (0.1, 5, 60) else 'FAIL'} одинаковый hit@6 → текущие настройки не трогаем")

    # если текущий вариант теряет контекст — переключаемся на лучший с рабочим весом
    rows = [row(0.0, 5, 60, 0.88, 1.0), row(0.1, 5, 60, 0.70, 0.88), row(0.05, 5, 60, 0.82, 1.0)]
    w, d, k, m, why = choose(rows, current=(0.1, 5, 60))
    ok &= (w, d) == (0.05, 5)
    print(f"  {'OK  ' if (w, d) == (0.05, 5) else 'FAIL'} текущий теряет контекст → берём лучший с весом ≥ 0.05")

    # вес 0 при прочих равных не выбираем
    rows = [row(0.0, 5, 60, 0.88, 1.0), row(0.05, 5, 60, 0.88, 1.0)]
    w, d, k, m, why = choose(rows, current=(0.3, 5, 60))
    ok &= w >= MIN_USEFUL_WEIGHT
    print(f"  {'OK  ' if w >= MIN_USEFUL_WEIGHT else 'FAIL'} при равных метриках вес 0 не берём (выбрано {w})")

    # RRF_K различает варианты: при равном hit@6 берём тот, где hit@1 выше
    rows = [row(0.1, 5, 60, 0.76, 1.0), row(0.1, 5, 10, 0.88, 1.0)]
    w, d, k, m, why = choose(rows, current=(0.3, 5, 60))
    ok &= k == 10
    print(f"  {'OK  ' if k == 10 else 'FAIL'} при равном hit@6 выбираем RRF_K с лучшим hit@1 (выбрано {k})")

    print("  ИТОГ: логика выбора в порядке" if ok else "  ИТОГ: логика выбора сломана")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vector-k", type=int, default=50,
                        help="сколько кандидатов брать у векторов при замере")
    parser.add_argument("--questions", type=int, default=len(GOLDEN))
    parser.add_argument("--selftest", action="store_true",
                        help="проверить логику выбора без обращения к API")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    if not config.EMB_API_KEY:
        print("Не задан EMB_API_KEY — прописать ключ в .env")
        return 1

    docs = ingest.load_documents_from_zip(ingest.find_knowledge_base(config.DATA_DIR))
    chunks, _ = chunking.build_chunks(docs, verbose=False)
    lexical = search.build_index(chunks)

    questions = {
        row["id"]: row["question"]
        for row in csv.DictReader(open(config.DATA_DIR / "demo_questions.csv", encoding="utf-8-sig"))
    }

    client = make_client()
    try:
        vectors, rebuilt = build_or_load(chunks, client, model=config.EMB_MODEL)
        print(config.describe())
        print(f"Векторный индекс: {'собран заново' if rebuilt else 'взят из кэша'}\n")

        # ---------------------------------------------------------- данные по вопросам
        qids = [qid for qid in GOLDEN if qid in questions][: args.questions]
        prepared = {qid: query_prep.prepare_search_query(questions[qid]) for qid in qids}

        vector_ranking: dict[str, list[int]] = {}
        bm25_ranking: dict[str, list[int]] = {}
        bodies = [chunk.body for chunk in chunks]

        for qid in qids:
            text = prepared[qid]
            try:
                query_vector = client.embed_query(text)
            except LLMError as exc:
                print(f"Не удалось посчитать вектор запроса {qid}: {exc}")
                return 1
            vector_ranking[qid] = [
                hit.position for hit in vectors.search_vector(query_vector, k=args.vector_k)
            ]
            bm25_ranking[qid] = [
                hit.position for hit in lexical.search_tokens(textproc.tokenize_query(text), k=50)
            ]

        def evaluate(weight: float, depth: int | None, rrf_k: int) -> dict[str, float]:
            ranks: list[int | None] = []
            for qid in qids:
                fused = reciprocal_rank_fusion(
                    [bm25_ranking[qid], vector_ranking[qid]],
                    k=rrf_k,
                    weights=[weight, 1.0],
                    depths=[depth, None],
                )
                ordered = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
                ranks.append(rank_of([bodies[pos] for pos, _ in ordered], GOLDEN[qid]))
            return metrics(ranks)

        # ------------------------------------------------------------ опорные варианты
        only_bm25 = metrics([rank_of([bodies[p] for p in bm25_ranking[q]], GOLDEN[q]) for q in qids])
        only_vector = metrics([rank_of([bodies[p] for p in vector_ranking[q]], GOLDEN[q]) for q in qids])

        header = f"{'вес':>5s} {'глубина':>8s} {'RRF_K':>6s} {'hit@1':>6s} {'hit@5':>6s} {'hit@6':>6s} {'MRR':>6s}"
        print(f"{'вариант':26s} {'hit@1':>6s} {'hit@5':>6s} {'hit@6':>6s} {'MRR':>6s}")
        print(f"{'только BM25':26s} {only_bm25['hit@1']:6.2f} {only_bm25['hit@5']:6.2f} "
              f"{only_bm25[f'hit@{CONTEXT_K}']:6.2f} {only_bm25['mrr']:6.2f}")
        print(f"{'только векторы':26s} {only_vector['hit@1']:6.2f} {only_vector['hit@5']:6.2f} "
              f"{only_vector[f'hit@{CONTEXT_K}']:6.2f} {only_vector['mrr']:6.2f}")

        print("\n" + "=" * 74)
        print(f"СЛИЯНИЕ: вес × глубина × RRF_K   (решаем по hit@{CONTEXT_K})\n")
        print(header)
        print("-" * 74)

        grid_weights = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 1.0)
        grid_depths: tuple[int | None, ...] = (5, 10, None)
        grid_k = (10, 20, 40, 60)
        results: list[tuple[float, int | None, int, dict[str, float]]] = []
        for rrf_k in grid_k:
            for weight in grid_weights:
                for depth in grid_depths:
                    m = evaluate(weight, depth, rrf_k)
                    results.append((weight, depth, rrf_k, m))
                    is_current = (abs(weight - config.BM25_WEIGHT) < 1e-9
                                  and (depth or 0) == (config.BM25_FUSION_DEPTH or 0)
                                  and rrf_k == config.RRF_K)
                    mark = "  ← сейчас в конфиге" if is_current else ""
                    print(f"{weight:5.2f} {str(depth or 'все'):>8s} {rrf_k:6d} {m['hit@1']:6.2f} "
                          f"{m['hit@5']:6.2f} {m[f'hit@{CONTEXT_K}']:6.2f} {m['mrr']:6.2f}{mark}")

        # ------------------------------------------------------------------ выбор
        weight, depth, rrf_k, m, reason = choose(
            results, current=(config.BM25_WEIGHT, config.BM25_FUSION_DEPTH, config.RRF_K)
        )
        best_ctx = max(r[3][f"hit@{CONTEXT_K}"] for r in results)
        best_h1 = max(r[3]["hit@1"] for r in results)

        print("\n" + "=" * 74)
        print(f"Лучший hit@{CONTEXT_K} в сетке: {best_ctx:.2f} | лучший hit@1: {best_h1:.2f}")
        print(f"\nВыбрано: вес {weight}, глубина {depth or 'все'}, RRF_K {rrf_k} → "
              f"hit@1 {m['hit@1']:.2f}, hit@5 {m['hit@5']:.2f}, "
              f"hit@{CONTEXT_K} {m[f'hit@{CONTEXT_K}']:.2f}, MRR {m['mrr']:.2f}")
        print(f"Почему: {reason}")

        print("\nПропишите в .env:")
        print(f"  BM25_WEIGHT={weight}")
        print(f"  BM25_FUSION_DEPTH={depth or 30}")
        if rrf_k != config.RRF_K:
            print(f"  # RRF_K={rrf_k}  (по умолчанию 60; впишите, если хотите закрепить)")

        print("\nНапоминание: 17 вопросов — маленькая выборка, разница меньше 0.05 — шум.")
        print("Проверить логику выбора без сети: python tune_fusion.py --selftest")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
