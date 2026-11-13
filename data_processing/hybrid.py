"""Гибридный поиск: BM25 + векторы → RRF → (необязательно) реранкер.

Зачем объединять два поиска
===========================
Каждый ошибается по-своему:

  * BM25 силён на точных токенах: номера договоров, ИНН, суммы, номера пунктов.
    Слаб на перефразировках («грузовик ночью» вместо «доступ транспорта
    круглосуточно»).
  * Векторы сильны на смысле. Слабы на мелких различиях: «П-17/2024» и
    «ПТ-05/2023» для них почти одно и то же, а гарантия по этим договорам разная.

Объединение даёт лучший результат, чем любой из них по отдельности. Это не
догадка: на демо-наборе BM25 даёт hit@5 = 0.88, гибрид — выше (см. check_step4.py).

Что такое RRF и почему именно он
================================
Reciprocal Rank Fusion — способ сложить два ранжирования, не зная их шкал.
Оценки BM25 (десятки) и косинус (0…1) несопоставимы: складывать их напрямую
нельзя. RRF смотрит не на оценки, а на МЕСТА:

    балл(фрагмент) = Σ 1 / (k + место_в_списке)      по всем спискам, где он есть

k=60 сглаживает: разница между 1-м и 2-м местом меньше, чем между 1-м и 15-м.
Фрагмент, который оба поиска поставили высоко, получит большой балл. Фрагмент,
который нашёл только один поиск, — тоже получит, но меньше. Это ровно то
поведение, которое нужно.
"""

from __future__ import annotations

from dataclasses import dataclass

if __package__:
    from . import config, textproc
    from .chunking import Chunk
    from .llm_client import LLMClient, LLMError
    from .search import LexicalIndex
    from .vector_index import VectorIndex
else:
    import config
    import textproc
    from chunking import Chunk
    from llm_client import LLMClient, LLMError
    from search import LexicalIndex
    from vector_index import VectorIndex


@dataclass
class FusedHit:
    """Фрагмент после объединения двух поисков."""

    chunk: Chunk
    rrf_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rerank_score: float | None = None

    def describe(self) -> str:
        return (
            f"rrf={self.rrf_score:.4f}"
            f" | bm25={self.bm25_rank or '—'}"
            f" | vec={self.vector_rank or '—'}"
            + (f" | rerank={self.rerank_score:.3f}" if self.rerank_score is not None else "")
        )


def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = config.RRF_K,
    weights: list[float] | None = None,
    depths: list[int | None] | None = None,
) -> dict[int, float]:
    """Сложить несколько ранжирований (списки позиций фрагментов) в один балл.

    Пример: rankings = [[5, 2, 9], [2, 5, 1]] — BM25 поставил фрагмент 5 первым,
    а векторы — вторым. Фрагмент 2 наоборот. Оба получат высокий суммарный балл,
    и это правильно: два независимых метода на него указали.

    weights — вес каждого ранжирования (по умолчанию все равны 1.0).
    depths  — сколько первых мест учитывать (None — все). Нужно, когда поиск
              надёжен в начале списка и шумит в хвосте.

    Зачем неравные веса. Классический RRF предполагает, что методы одинаково
    хороши. У нас это не так: на демо-вопросах векторы дают hit@1 0.88, а BM25 —
    0.53. При равных весах BM25-лидер, который векторы поставили на 2-е место,
    обгоняет правильный фрагмент (1-е место у векторов, далёкое у BM25) — и
    качество падает относительно одного векторного поиска. Вес исправляет это
    честно: сильному методу больше доверия, слабый работает как уточнение.
    """
    count = len(rankings)
    weights = list(weights) if weights is not None else [1.0] * count
    depths = list(depths) if depths is not None else [None] * count
    if len(weights) != count or len(depths) != count:
        raise ValueError("weights и depths должны быть той же длины, что rankings")

    scores: dict[int, float] = {}
    for ranking, weight, depth in zip(rankings, weights, depths):
        for place, position in enumerate(ranking, start=1):
            if depth is not None and place > depth:
                break
            scores[position] = scores.get(position, 0.0) + weight / (k + place)
    return scores


class HybridSearcher:
    """Итоговый поиск решения: лексика + векторы + реранк."""

    def __init__(self, lexical: LexicalIndex, vectors: VectorIndex, client: LLMClient | None = None) -> None:
        if len(lexical) != len(vectors):
            raise ValueError(
                f"Индексы не совпадают: BM25 по {len(lexical)} фрагментам, векторы по {len(vectors)}"
            )
        self.lexical = lexical
        self.vectors = vectors
        self.client = client

    @property
    def chunks(self) -> list[Chunk]:
        return self.vectors.chunks

    # ------------------------------------------------------------- поиск
    def search(self, query_text: str, *, final_k: int = 6, use_rerank: bool = True) -> list[FusedHit]:
        """Ищет один вопрос, переиспользуя общий путь пакетного поиска."""
        candidates = self.search_many([query_text], final_k=max(final_k, config.RERANK_CANDIDATES))[0]
        if use_rerank:
            candidates = self.rerank_candidates(query_text, candidates)
        return candidates[:final_k]

    def search_many(self, queries: list[str], *, final_k: int = 30) -> list[list[FusedHit]]:
        """Кодирует формулировки одним батчем и возвращает списки без реранжа."""
        query_vectors = [None] * len(queries)
        if self.client is not None:
            try:
                texts = [config.EMB_QUERY_PREFIX + query for query in queries]
                query_vectors = self.client.embed(texts)
                if len(query_vectors) != len(queries):
                    raise ValueError("Число векторов не совпадает с числом запросов")
            except Exception as exc:
                print(f"[hybrid] эмбеддинги запросов недоступны ({type(exc).__name__}); BM25")
                query_vectors = [None] * len(queries)
        rankings = []
        for query, vector in zip(queries, query_vectors):
            lexical_hits = self.lexical.search(query, k=config.BM25_TOP_K)
            try:
                vector_hits = self.vectors.search_vector(vector, k=config.VECTOR_TOP_K) if vector is not None else []
            except Exception as exc:
                print(f"[hybrid] векторный поиск недоступен ({type(exc).__name__}); BM25")
                vector_hits = []
            rankings.append(self._fuse(lexical_hits, vector_hits)[:final_k])
        return rankings

    def _fuse(self, lexical_hits: list, vector_hits: list) -> list[FusedHit]:
        """Объединяет сигналы; без векторов сохраняет всю лексическую выдачу."""
        fused_scores = reciprocal_rank_fusion(
            [[hit.position for hit in lexical_hits], [hit.position for hit in vector_hits]],
            weights=[config.BM25_WEIGHT if vector_hits else 1.0, 1.0],
            depths=[config.BM25_FUSION_DEPTH if vector_hits else None, None],
        )
        bm25_map = {hit.position: hit for hit in lexical_hits}
        vector_map = {hit.position: hit for hit in vector_hits}
        vector_ranks = {hit.position: rank for rank, hit in enumerate(vector_hits, start=1)}
        candidates: list[FusedHit] = []
        for position, score in fused_scores.items():
            candidates.append(
                FusedHit(
                    chunk=self.chunks[position],
                    rrf_score=score,
                    bm25_rank=bm25_map[position].rank if position in bm25_map else None,
                    vector_rank=vector_ranks.get(position),
                    bm25_score=bm25_map[position].score if position in bm25_map else 0.0,
                    vector_score=vector_map[position].score if position in vector_map else 0.0,
                )
            )
        candidates.sort(key=lambda hit: hit.rrf_score, reverse=True)

        return candidates

    def rerank_candidates(self, query_text: str, candidates: list[FusedHit]) -> list[FusedHit]:
        """Ранжирует объединённую выдачу по исходному смыслу вопроса один раз."""
        if self.client is not None and len(candidates) > 1:
            try:
                candidates = self._rerank(query_text, candidates)
            except Exception as exc:      # noqa: BLE001 — последний рубеж защиты
                print(f"[hybrid] реранж отключён из-за ошибки ({type(exc).__name__}); порядок RRF")
        return candidates

    # ------------------------------------------------------------ реранк
    def _rerank(self, query_text: str, candidates: list[FusedHit]) -> list[FusedHit]:
        """Переранжировать кандидатов. Три режима, с деградацией по цепочке.

        endpoint → LLM → без реранжа. Любая ошибка на шаге означает переход к
        следующему, а не падение прогона: реранж улучшает качество, но не должен
        ломать работу.
        """
        head = candidates[: config.RERANK_CANDIDATES]
        tail = candidates[config.RERANK_CANDIDATES:]
        if len(head) < 2:
            return candidates

        documents = [textproc.evidence_text(hit.chunk.text) for hit in head]
        mode = config.RERANK_MODE
        if mode == "auto":
            mode = "endpoint" if config.RERANK_MODEL else "llm"

        ordered: list[tuple[int, float]] = []
        if mode == "endpoint":
            try:
                ordered = self.client.rerank(query_text, documents, top_n=config.RERANK_TOP_N)
            except Exception as exc:      # noqa: BLE001 — реранк не должен ронять прогон
                # Ловим ЛЮБУЮ ошибку, а не только LLMError: клиент может бросить
                # что угодно (ValueError при разборе ответа, ошибку SDK, таймаут).
                # Реранж — улучшение качества, а не обязательный шаг: при сбое
                # продолжаем с порядком RRF.
                print(f"[hybrid] /rerank недоступен ({type(exc).__name__}); пробую реранж через chat")
                mode = "llm"

        if mode == "llm":
            try:
                ordered = self.client.rerank_by_llm(
                    query_text, documents, top_n=config.RERANK_TOP_N
                )
            except Exception as exc:      # noqa: BLE001 — см. комментарий выше
                print(f"[hybrid] реранж через chat недоступен ({type(exc).__name__}: {exc}); остаётся порядок RRF")
                ordered = []

        if not ordered:
            return candidates

        reranked: list[FusedHit] = []
        used: set[int] = set()
        for index, score in ordered:
            if index in used:
                continue
            used.add(index)
            hit = head[index]
            hit.rerank_score = score
            reranked.append(hit)

        # Фрагменты, которые реранкер не взял, но которые были в RRF-выдаче,
        # дописываем в конец: иногда реранкер ошибается и «теряет» нужное.
        for index, hit in enumerate(head):
            if index not in used:
                reranked.append(hit)
        return reranked + tail

    # --------------------------------------------------------- удобные обёртки
    def best_bm25_score(self, query_text: str) -> float:
        """Лучшая оценка BM25 — для порога отказа на шаге 6."""
        return self.lexical.max_score(query_text)

    def chunks_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Найти фрагменты по их идентификаторам (для доп. соглашения и соседей)."""
        index_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        return [index_by_id[cid] for cid in chunk_ids if cid in index_by_id]
