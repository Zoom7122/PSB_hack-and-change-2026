"""Утилита для проверки поиска вручную.

Запуск:
    python3 probe.py "какая арендная плата по договору с Северным Складом"
    python3 probe.py                      # интерактивный режим: вводите вопросы подряд
    python3 probe.py --k 10 --explain "испытательный срок главного бухгалтера"

Зачем нужна отдельная утилита:
  это ваш «микроскоп». Когда на шагах 5–7 ответы будут плохими, первый вопрос
  всегда один: «поиск не нашёл нужное или модель не смогла сформулировать?».
  probe.py отвечает на первую половину за секунду, без вызовов модели и без
  траты лимита сабмитов.
"""

from __future__ import annotations

import argparse

import chunking
import config
import ingest
import query_prep
import search


def build() -> search.LexicalIndex:
    """Прочитать базу, нарезать фрагменты и построить BM25."""
    docs = ingest.load_documents_from_zip(ingest.find_knowledge_base(config.DATA_DIR))
    chunks, _ = chunking.build_chunks(docs, verbose=False)
    return search.build_index(chunks)


def show(index: search.LexicalIndex, question: str, k: int, explain: bool) -> None:
    """Напечатать результат поиска по одному вопросу."""
    prepared_query = query_prep.prepare_search_query(question)
    tokens = query_prep.prepare_search_tokens(question)

    print("=" * 100)
    if prepared_query != question.strip():
        print(f"ВОПРОС (диалог): {question.strip()[:200]}")
        print(f"ПОИСК ПО:        {prepared_query}")
    else:
        print(f"ВОПРОС: {question.strip()}")
    print(f"ТОКЕНЫ: {tokens[:14]}{' …' if len(tokens) > 14 else ''}")
    print()

    hits = index.search_tokens(tokens, k=k)
    if not hits:
        print("Ничего не найдено: ни один фрагмент не содержит слов запроса.")
        print("Это сигнал, что база не покрывает тему — на шаге 6 такой вопрос "
              "должен приводить к обоснованному отказу.")
        return

    for hit in hits:
        chunk = hit.chunk
        print(f"{hit.rank}. оценка {hit.score:6.2f} | {chunk.label:14s} | {chunk.section[:40]}")
        print(f"   документ: {chunk.doc_id}")
        body = " ".join(chunk.body.split())
        print(f"   текст:    {body[:260]}{'…' if len(body) > 260 else ''}")
        if explain:
            matched = [t for t in tokens if t in index.corpus_tokens[hit.position]]
            print(f"   совпавшие токены: {sorted(set(matched))[:12]}")
        print()

    print(f"Лучшая оценка: {hits[0].score:.2f}  (пригодится для порога отказа на шаге 6)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка лексического поиска")
    parser.add_argument("question", nargs="*", help="вопрос; без него — интерактивный режим")
    parser.add_argument("--k", type=int, default=5, help="сколько фрагментов показать")
    parser.add_argument("--explain", action="store_true", help="показать совпавшие токены")
    args = parser.parse_args()

    index = build()
    print(f"Индекс: {len(index)} фрагментов\n")

    if args.question:
        show(index, " ".join(args.question), args.k, args.explain)
        return

    print("Интерактивный режим. Пустая строка или Ctrl+C — выход.\n")
    while True:
        try:
            question = input("вопрос> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        show(index, question, args.k, args.explain)


if __name__ == "__main__":
    main()
