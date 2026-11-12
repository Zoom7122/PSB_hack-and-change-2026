"""Шаг 3: лексический поиск по фрагментам (BM25).

Что такое BM25 и зачем он нужен
===============================
BM25 — это «умный полнотекстовый поиск». Он оценивает релевантность фрагмента
запросу по трём соображениям:

  1) чем чаще слово запроса встречается во фрагменте — тем выше оценка (TF);
  2) чем реже слово встречается во всей базе — тем оно важнее (IDF):
     «арендная» важно, «договор» встречается везде и почти ничего не значит;
  3) чем длиннее фрагмент, тем ниже оценка за то же число совпадений, иначе
     длинные фрагменты выигрывали бы всегда.

Два параметра настраивают это поведение: k1 (насыщение по частоте) и b
(нормализация по длине). Значения по умолчанию k1=1.5, b=0.75 — стандарт,
на небольших корпусах их почти никогда не меняют.

Почему BM25 обязателен в юридическом поиске
===========================================
Векторный поиск (шаг 4) хорош на перефразировках, но плохо различает точные
номера: «П-17/2024» и «ПТ-05/2023» для него почти одинаковые строки. А для
юриста разница между договорами — это разница между «гарантия 24 месяца» и
«гарантия по талону производителя». BM25 такие токены различает идеально.
Поэтому на шаге 4 мы объединим оба поиска, а не заменим один другим.

Чего BM25 не умеет (и это нормально на данном шаге)
==================================================
* синонимы: «сколько платим за склад» → «арендная плата» он не поймёт
  (частично лечится нашим SYNONYMS в textproc.py);
* смысл: «за сколько дней проверить качество» ≠ «приёмка по качеству»;
* отрицания и уточнения.

Именно поэтому на шаге 4 добавляются векторы и реранкер, а на шаге 5 —
понимание вопроса. Здесь мы строим базу и измеряем, сколько вытягивает
чистая лексика.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

import textproc
from chunking import Chunk

# Стандартные параметры BM25. Вынесены в константы, чтобы на шаге 4 можно было
# проверить на метриках, помогает ли их изменение.
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class Hit:
    """Результат поиска: фрагмент + его оценка + место в выдаче."""

    chunk: Chunk
    score: float
    rank: int           # 1 = лучший
    position: int = -1  # индекс фрагмента в общем списке (для отладки)

    @property
    def short(self) -> str:
        """Строка для логов и отладки."""
        return f"#{self.rank} {self.score:6.2f} | {self.chunk.label:12s} | {self.chunk.doc_id[:34]}"


class LexicalIndex:
    """Индекс BM25 поверх фрагментов базы знаний.

    Пример использования:
        index = LexicalIndex(chunks)
        hits = index.search("какая арендная плата по договору с Северным Складом", k=5)
        for hit in hits:
            print(hit.short, hit.chunk.body[:80])
    """

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B) -> None:
        if not chunks:
            raise ValueError("Нельзя построить индекс по пустому списку фрагментов")
        self.chunks = chunks
        # Токенизируем именно chunk.text — то есть шапку вместе с телом.
        # Шапка содержит номер документа, ИНН и дату, и именно по ним юрист
        # часто формулирует запрос («что сказано в договоре П-17/2024»).
        self.corpus_tokens: list[list[str]] = [
            textproc.tokenize_document(chunk.text) for chunk in chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens, k1=k1, b=b)
        self.k1 = k1
        self.b = b

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Найти top-k фрагментов по тексту запроса."""
        return self.search_tokens(textproc.tokenize_query(query), k=k)

    def search_tokens(self, tokens: list[str], k: int = 5) -> list[Hit]:
        """Найти top-k фрагментов по запросу.

        Возвращаем только фрагменты с положительной оценкой: у BM25 нулевая
        оценка означает «ни одного общего токена», и такие фрагменты в контекст
        отдавать нельзя — модель начнёт отвечать по случайному документу.
        """
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        # Сортируем по убыванию оценки. Используем enumerate, чтобы сохранить
        # ссылку на исходный фрагмент: BM25 возвращает только числа.
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

        hits: list[Hit] = []
        for position, score in ranked:
            if score <= 0:
                break
            hits.append(
                Hit(
                    chunk=self.chunks[position],
                    score=float(score),
                    rank=len(hits) + 1,
                    position=position,
                )
            )
            if len(hits) >= k:
                break
        return hits

    def max_score(self, query: str) -> float:
        """Лучшая оценка по запросу — понадобится для решения «отказаться» (шаг 6)."""
        tokens = textproc.tokenize_query(query)
        if not tokens:
            return 0.0
        return float(max(self.bm25.get_scores(tokens)))

    def explain(self, query: str, k: int = 5) -> str:
        """Подробный разбор: токены запроса и что с чем совпало.

        Нужно, когда поиск промахивается: сразу видно, проблема в токенизации
        (слово приняло неожиданную форму) или в самом документе (нужного слова
        там действительно нет).
        """
        tokens = textproc.tokenize_query(query)
        lines = [f"Запрос: {query}", f"Токены: {tokens}", ""]
        for hit in self.search(query, k=k):
            matched = [t for t in tokens if t in self.corpus_tokens[hit.position]]
            lines.append(f"{hit.short}")
            lines.append(f"    совпало: {matched}")
            lines.append(f"    {hit.chunk.body[:140].replace(chr(10), ' ')}")
        return "\n".join(lines)


def build_index(chunks: list[Chunk]) -> LexicalIndex:
    """Удобная обёртка, чтобы вызывающий код не думал о параметрах."""
    return LexicalIndex(chunks)
