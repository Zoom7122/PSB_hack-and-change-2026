"""Самопроверка шага 4: векторный индекс, RRF и гибридный поиск.

Запуск:
    python3 check_step4.py            # только офлайновые проверки (без сети)
    python3 check_step4.py --online   # плюс реальные эмбеддинги и метрики качества

Почему проверок два режима
==========================
Механика (матрица, нормировка, RRF, сборка/загрузка индекса, деградация без
сети) проверяется на синтетических векторах — это быстро и не требует ключей.
Качество (hit@k на демо-вопросах) проверить синтетикой нельзя: нужны настоящие
эмбеддинги. Поэтому второй режим требует настроенный API.

Синтетические векторы НЕ измеряют качество. Они дают «случайный, но
воспроизводимый» шум, чтобы проверить, что код не падает и что RRF считает
правильно.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys

import numpy as np

import chunking
import config
import ingest
import query_prep
import search
import textproc
from hybrid import HybridSearcher, reciprocal_rank_fusion
from vector_index import VectorIndex, build_or_load

GOLDEN: dict[str, str] = {
    "1": "450 000", "2": "шести месяцев", "4": "450 000", "5": "10 000 000",
    "6": "5 (пяти) рабочих дней", "7": "круглосуточно", "8": "акта о выявленных недостатках",
    "9": "допускается предоплата", "10": "Инициатором согласования",
    "11": "не более 3 (трёх) рабочих дней", "12": "распространяются положения трудового законодательства",
    "13": "может быть отозвано", "14": "сезонных работ", "15": "по акту в состоянии",
    "16": "стоимость упаковки и доставки", "17": "гарантийном талоне",
    "18": "в течение 3 (трёх) рабочих дней",
}


class FakeEmbedder:
    """Псевдо-эмбеддер: детерминированный вектор из хеша текста.

    Нужен только чтобы проверить механику без сети. Качество такого «поиска»
    бессмысленно — он не знает ни слов, ни смысла.
    """

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        return self.embed(texts, batch_size)

    def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        self.calls += 1
        result = []
        for text in texts:
            seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            result.append(rng.normal(size=self.dim).tolist())
        return result

    def chat(self, *args, **kwargs):        # noqa: D401 - заглушка
        raise RuntimeError("FakeEmbedder не умеет chat")

    def rerank(self, *args, **kwargs):
        raise RuntimeError("FakeEmbedder не умеет rerank")

    def rerank_by_llm(self, *args, **kwargs):
        raise RuntimeError("FakeEmbedder не умеет rerank_by_llm")


def metrics(ranks: dict[str, int | None]) -> dict[str, float]:
    """hit@k и MRR по словарю «вопрос → место правильного фрагмента»."""
    total = len(ranks)
    result = {}
    # hit@6 — не «ещё одна метрика»: ровно столько фрагментов уходит в промпт
    # (final_k=6). Именно она отвечает на вопрос «увидит ли модель нужный текст».
    for k in (1, 3, 5, 6, 10):
        result[f"hit@{k}"] = sum(1 for r in ranks.values() if r and r <= k) / total
    result["mrr"] = sum(1 / r for r in ranks.values() if r) / total
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="считать hit@1 жёстким критерием (по умолчанию — предупреждение)")
    parser.add_argument("--online", action="store_true", help="проверить на реальных эмбеддингах")
    args = parser.parse_args()

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  → {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    docs = ingest.load_documents_from_zip(ingest.find_knowledge_base(config.DATA_DIR))
    chunks, _ = chunking.build_chunks(docs, verbose=False)
    lexical = search.build_index(chunks)

    # ------------------------------------------------------------------ A. RRF
    print("A. ОБЪЕДИНЕНИЕ РАНЖИРОВАНИЙ (RRF)")
    fused = reciprocal_rank_fusion([[5, 2, 9], [2, 5, 1]])
    check(fused[2] > fused[9], "фрагмент, который оба поиска нашли, весит больше найденного одним")
    check(abs(fused[2] - fused[5]) < 1e-9, "фрагменты с зеркальными местами получают равный балл")
    check(reciprocal_rank_fusion([]) == {}, "пустой список ранжирований даёт пустой балл")
    check(len(reciprocal_rank_fusion([[1, 2, 3]])) == 3, "одно ранжирование сохраняет все элементы")

    # взвешенное слияние: вес 0 у первого списка = его голос не учитывается
    weighted = reciprocal_rank_fusion([[5, 2, 9], [2, 5, 1]], weights=[0.0, 1.0])
    check(abs(weighted[2] - 1.0 / (config.RRF_K + 1)) < 1e-9
          and sorted(weighted, key=lambda p: -weighted[p])[:2] == [2, 5],
          "вес 0 полностью выключает голос одного поиска")
    # глубина: места дальше N не учитываются
    capped = reciprocal_rank_fusion([[1, 2, 3, 4]], depths=[2])
    check(3 not in capped and 4 not in capped, "глубина отсекает хвост ранжирования")
    # при весе 0.3 сильный поиск обгоняет слабый (ровно наш случай: векторы 0.88, BM25 0.53)
    strong_wins = reciprocal_rank_fusion([[5, 2], [2, 5]], weights=[0.3, 1.0])
    check(strong_wins[2] > strong_wins[5], "при неравных весах выигрывает сильный поиск")

    # ------------------------------------------------------ B. ВЕКТОРНЫЙ ИНДЕКС
    print("\nB. ВЕКТОРНЫЙ ИНДЕКС (синтетические векторы)")
    fake = FakeEmbedder(dim=64)
    from vector_index import build_vector_index

    vectors = build_vector_index(chunks, fake, verbose=False)
    check(len(vectors) == len(chunks), f"векторов столько же, сколько фрагментов ({len(vectors)})")
    check(vectors.dim == 64, f"размерность взята из модели ({vectors.dim})")
    norms = np.linalg.norm(vectors.matrix, axis=1)
    check(np.allclose(norms, 1.0, atol=1e-4), "все векторы нормированы к единичной длине")

    hits = vectors.search_vector(fake.embed(["арендная плата"])[0], k=5)
    check(len(hits) == 5, "поиск по вектору возвращает 5 результатов")
    check(all(-1.01 <= h.score <= 1.01 for h in hits), "косинусное сходство лежит в пределах [-1, 1]")
    check(hits[0].score >= hits[-1].score, "оценки убывают")

    try:
        vectors.search_vector([0.0] * 128, k=3)
        check(False, "чужая размерность запроса отклоняется")
    except ValueError:
        check(True, "чужая размерность запроса отклоняется")

    # -------------------------------------------------- C. СОХРАНЕНИЕ ИНДЕКСА
    print("\nC. КЭШ ИНДЕКСА")
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        vectors.save(tmp, model="fake-model")
        loaded = VectorIndex.load(chunks, tmp, model="fake-model")
        check(np.allclose(loaded.matrix, vectors.matrix), "индекс сохраняется и загружается без искажений")

        try:
            VectorIndex.load(chunks, tmp, model="другая-модель")
            check(False, "индекс другой модели отклоняется")
        except ValueError:
            check(True, "индекс другой модели отклоняется (защита от несовпадения размерностей)")

        try:
            VectorIndex.load(chunks[:-5], tmp, model="fake-model")
            check(False, "индекс для другого набора фрагментов отклоняется")
        except ValueError:
            check(True, "индекс для другого набора фрагментов отклоняется")

        loaded_again, rebuilt = build_or_load(chunks, fake, model="fake-model", directory=tmp)
        check(not rebuilt, "build_or_load берёт готовый индекс из кэша, не пересчитывая")
        (Path(tmp) / "vector_index.npz").unlink()
        _, rebuilt = build_or_load(chunks, fake, model="fake-model", directory=tmp)
        check(rebuilt, "при отсутствии кэша индекс пересобирается")

    # ------------------------------------------------------- D. ГИБРИДНЫЙ ПОИСК
    print("\nD. ГИБРИДНЫЙ ПОИСК")
    hybrid = HybridSearcher(lexical, vectors, client=fake)
    results = hybrid.search("арендная плата по договору с Северным Складом", final_k=6, use_rerank=False)
    check(len(results) == 6, f"возвращается final_k фрагментов ({len(results)})")
    check(all(r.chunk is not None for r in results), "у каждого результата есть фрагмент")
    check(any(r.bm25_rank and r.vector_rank for r in results),
          "есть фрагменты, найденные обоими поисками (значит RRF реально объединяет)")
    check(all(r.rrf_score > 0 for r in results), "у всех результатов положительный RRF-балл")

    no_vectors = HybridSearcher(lexical, vectors, client=None)
    fallback = no_vectors.search("арендная плата", final_k=5)
    check(len(fallback) == 5, "без клиента эмбеддингов поиск продолжает работать на одном BM25")

    class BrokenEmbedder(FakeEmbedder):
        def embed(self, texts, batch_size=None):
            from llm_client import LLMError
            raise LLMError("сеть недоступна")

    broken = HybridSearcher(lexical, vectors, client=BrokenEmbedder())
    degraded = broken.search("арендная плата", final_k=5)
    check(len(degraded) == 5, "при падении эмбеддингов поиск деградирует, а не падает")

    # -------------------------------------------------------- E. КАЧЕСТВО (сеть)
    if not args.online:
        print("\nE. КАЧЕСТВО — пропущено (запустите с --online, нужен настроенный API)")
    else:
        print("\nE. КАЧЕСТВО НА РЕАЛЬНЫХ ЭМБЕДДИНГАХ")
        from llm_client import make_client

        print(config.describe())
        client = make_client()
        try:
            vectors_real, rebuilt = build_or_load(chunks, client, model=config.EMB_MODEL)
            if rebuilt:
                print("(индекс собран заново и сохранён в artifacts/)")

            questions = {
                row["id"]: row["question"]
                for row in csv.DictReader(open(config.DATA_DIR / "demo_questions.csv", encoding="utf-8-sig"))
            }

            def evaluate(finder) -> dict[str, int | None]:
                ranks: dict[str, int | None] = {}
                for qid, needle in GOLDEN.items():
                    prepared = query_prep.prepare_search_query(questions[qid])
                    found = finder(prepared)
                    ranks[qid] = None
                    for place, chunk in enumerate(found, start=1):
                        if needle in chunk.body:
                            ranks[qid] = place
                            break
                return ranks

            # Запрос уже подготовлен выше (prepare_search_query). Повторно
            # вызывать prepare_search_tokens НЕЛЬЗЯ: стемминг, применённый дважды,
            # превращает основы в мусор — на вопросе 18 правильный фрагмент уезжал
            # с 4-го места на 99-е, и BM25 в отчёте выглядел хуже, чем есть.
            # Берём ровно тот путь, что и HybridSearcher: tokenize_query.
            bm25_ranks = evaluate(
                lambda q: [hit.chunk for hit in lexical.search_tokens(
                    textproc.tokenize_query(q), k=50)]
            )
            # embed_query, а не embed: у моделей с инструкцией запрос
            # кодируется иначе, чем документ (см. config.EMB_QUERY_PREFIX)
            embed_query = getattr(client, "embed_query", None)
            encode = (lambda text: embed_query(text)) if embed_query else (
                lambda text: client.embed([text])[0])

            vector_ranks = evaluate(
                lambda q: [hit.chunk for hit in vectors_real.search_vector(encode(q), k=50)]
            )
            hybrid_searcher = HybridSearcher(lexical, vectors_real, client=client)
            # use_rerank=False — иначе строка «гибрид (RRF)» незаметно включала бы
            # реранж и мы сравнивали бы не те варианты
            hybrid_ranks = evaluate(
                lambda q: [hit.chunk for hit in hybrid_searcher.search(q, final_k=50, use_rerank=False)]
            )
            full_ranks = evaluate(
                lambda q: [hit.chunk for hit in hybrid_searcher.search(q, final_k=50, use_rerank=True)]
            )

            print(f"\n{'вариант':22s} {'hit@1':>7s} {'hit@5':>7s} {'hit@10':>7s} {'MRR':>7s}")
            for name, ranks in (
                ("только BM25", bm25_ranks),
                ("только векторы", vector_ranks),
                ("гибрид (RRF)", hybrid_ranks),
                ("гибрид + реранк", full_ranks),
            ):
                m = metrics(ranks)
                print(f"{name:22s} {m['hit@1']:7.2f} {m['hit@5']:7.2f} {m['hit@10']:7.2f} {m['mrr']:7.2f}")

            hybrid_m = metrics(hybrid_ranks)
            bm25_m = metrics(bm25_ranks)
            vector_m = metrics(vector_ranks)

            # Главный критерий: гибрид не должен проигрывать ЛУЧШЕМУ из двух поисков.
            # Сравнивать только с BM25 мало: если векторы сильнее, «гибрид не хуже
            # BM25» пройдёт, а слияние при этом будет вредить.
            best_single = max(bm25_m, vector_m, key=lambda m: (m["hit@5"], m["hit@1"]))
            check(hybrid_m["hit@5"] >= best_single["hit@5"] - 0.02,
                  f"гибрид не хуже лучшего из двух по hit@5 "
                  f"({hybrid_m['hit@5']:.2f} против {best_single['hit@5']:.2f})")
            # hit@6 — жёсткий критерий: столько фрагментов уходит в промпт.
            check(hybrid_m["hit@6"] >= best_single["hit@6"] - 0.02,
                  f"гибрид не теряет попадание в контекст по hit@6 "
                  f"({hybrid_m['hit@6']:.2f} против {best_single['hit@6']:.2f})")

            # hit@1 — порядок ВНУТРИ этих шести фрагментов. На ответ влияет слабо
            # (модель видит все шесть), и выправляется реранжером на платформе,
            # поэтому по умолчанию это предупреждение, а не провал.
            # Хотите строгий режим с гейтом по hit@1 — запустите с --strict.
            gap = best_single["hit@1"] - hybrid_m["hit@1"]
            if gap > 0.05:
                message = (f"гибрид ставит правильный фрагмент первым реже "
                           f"({hybrid_m['hit@1']:.2f} против {best_single['hit@1']:.2f}, "
                           f"разрыв {gap:.2f})")
                if args.strict:
                    check(False, message)
                else:
                    print(f"ВНИМАНИЕ  {message}")
                    print("          На ответ не влияет: в промпт уходят все 6 фрагментов,")
                    print("          а порядок выправит реранжер. Поднять hit@1 можно через")
                    print("          RRF_K в tune_fusion.py (меньше K — резче разделение мест).")
            else:
                check(True, f"гибрид не размывает первое место по hit@1 "
                            f"({hybrid_m['hit@1']:.2f} против {best_single['hit@1']:.2f})")
            check(hybrid_m["hit@5"] >= 0.90, f"гибрид hit@5 ≥ 0.90 (получено {hybrid_m['hit@5']:.2f})")
            if hybrid_m["hit@5"] < best_single["hit@5"]:
                print(f"\n  Совет: слияние тянет вниз. Подберите вес командой "
                      f"python tune_fusion.py\n  сейчас BM25_WEIGHT={config.BM25_WEIGHT}, "
                      f"BM25_FUSION_DEPTH={config.BM25_FUSION_DEPTH}")
        finally:
            client.close()

    print()
    if failures:
        print(f"ИТОГ: не прошло проверок — {len(failures)}. Шаг 4 не закрыт.")
        return 1
    print("ИТОГ: офлайновые проверки пройдены.")
    if not args.online:
        print("Для метрик качества запустите: python3 check_step4.py --online")
    return 0


if __name__ == "__main__":
    sys.exit(main())
