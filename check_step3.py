"""Самопроверка шага 3: токенизация, стеммер, BM25 и качество поиска.

Запуск:
    python3 check_step3.py

Три блока:
  A. ТЕКСТ — токенизатор и стеммер ведут себя как задумано (это юнит-тесты).
  B. ИНДЕКС — BM25 построен, оценки корректны, поиск не возвращает мусор.
  C. КАЧЕСТВО — на 17 демо-вопросах из набора организаторов считаем hit@1/3/5/10
     и MRR: попал ли фрагмент с правильным ответом в топ выдачи.

Почему качество тоже проверяется автоматически:
  с этого шага у нас есть число, которое можно улучшать. Любое изменение
  (стеммер, синонимы, размер фрагмента) должно его повышать, а не понижать.
  Без такой метрики тюнинг поиска превращается в гадание.
"""

from __future__ import annotations

import csv
import sys

import chunking
import config
import ingest
import query_prep
import search
import textproc

# Эталон для оценки качества: id вопроса -> фраза, которая обязана быть
# в правильном фрагменте. Фразы взяты из демо-ответов организаторов
# (data/demo_output.csv), поэтому они заведомо юридически точные.
# Вопроса №3 здесь нет: в базе нет данных о пособиях по беременности и родам,
# правильный ответ на него — обоснованный отказ.
GOLDEN: dict[str, str] = {
    "1": "450 000",
    "2": "шести месяцев",
    "4": "450 000",
    "5": "10 000 000",
    "6": "5 (пяти) рабочих дней",
    "7": "круглосуточно",
    "8": "акта о выявленных недостатках",
    "9": "допускается предоплата",
    "10": "Инициатором согласования",
    "11": "не более 3 (трёх) рабочих дней",
    "12": "распространяются положения трудового законодательства",
    "13": "может быть отозвано",
    "14": "сезонных работ",
    "15": "по акту в состоянии",
    "16": "стоимость упаковки и доставки",
    "17": "гарантийном талоне",
    "18": "в течение 3 (трёх) рабочих дней",
}

# Вопросы, на которых лексический поиск обязан быть точным: совпадение
# почти буквальное. Если они не в топ-1, сломан стеммер или токенизатор.
MUST_BE_TOP1 = {"2", "8", "11", "9", "16", "17"}

QUESTION_WITHOUT_ANSWER = "Какой размер пособия по беременности и родам?"


# Вывод в консоль: если поток перенаправлен (pipe/файл), Windows использует
# кодировку локали (cp1251/cp866), и печать символов «→», «×» роняет скрипт
# с UnicodeEncodeError. Переключаем такой поток на UTF-8; интерактивную
# консоль не трогаем, чтобы в ней осталась родная кодировка.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  → {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    # ------------------------------------------------------------------ A. ТЕКСТ
    print("A. ТОКЕНИЗАЦИЯ И СТЕММЕР")
    check(textproc.stem("поставки") == "поставк", "стеммер: «поставки» → «поставк»", textproc.stem("поставки"))
    check(textproc.stem("акту") == textproc.stem("акта") == "акт", "стеммер: «акту» и «акта» → «акт»")
    check(textproc.stem("рублей") == "рубл", "стеммер: «рублей» → «рубл»")
    check(textproc.stem("арендная") == "арендн", "стеммер: «арендная» → «арендн»")
    check("аренд" in textproc.tokenize("Аренды"), "стеммер: «Аренды» → «аренд»")
    check(textproc.tokenize("внесён") == textproc.tokenize("внесен"), "ё и е дают один токен")
    check("3.1" in textproc.tokenize("п. 3.1 договора"), "составной номер «3.1» сохранён одним токеном")
    check("какая" not in textproc.tokenize("какая арендная плата"), "служебные слова отфильтрованы")
    check("предоплат" in textproc.tokenize_query("аванс"), "синоним: «аванс» расширяется до «предоплат»")
    check(textproc.tokenize("") == [], "пустая строка даёт пустой список токенов")

    # --------------------------------------------------------------- B. ИНДЕКС
    print("\nB. ИНДЕКС BM25")
    docs = ingest.load_documents_from_zip(ingest.find_knowledge_base(config.DATA_DIR))
    chunks, _ = chunking.build_chunks(docs, verbose=False)
    index = search.build_index(chunks)

    check(len(index) == len(chunks) == 360, f"индекс построен по всем фрагментам ({len(index)})")
    check(all(index.corpus_tokens[i] for i in range(len(index))), "у каждого фрагмента есть токены")

    hits = index.search("какая арендная плата по договору с Северным Складом", k=5)
    check(len(hits) == 5, f"по запросу возвращается 5 фрагментов (получено {len(hits)})")
    check(all(h.score > 0 for h in hits), "в выдачу попадают только фрагменты с положительной оценкой")
    check([h.rank for h in hits] == [1, 2, 3, 4, 5], "ранги идут по порядку 1..5")
    check(hits[0].score >= hits[-1].score, "оценки убывают")

    arenda_hits = index.search("арендная плата Северный Склад", k=5)
    check(any(h.chunk.body.startswith("Арендная плата составляет 450 000") for h in arenda_hits),
          "«арендная плата Северный Склад» → пункт 3.1 в топ-5",
          "; ".join(h.chunk.label for h in arenda_hits))

    check(index.search("", k=5) == [], "пустой запрос не возвращает ничего")
    check(index.search("абракадабра цвиркшвабра", k=5) == [], "запрос без общих слов с базой ничего не возвращает")

    long_hits = index.search("договор", k=50)
    check(len(long_hits) == 50, f"параметр k соблюдается (запрошено 50, получено {len(long_hits)})")

    # ------------------------------------------------------------ C. КАЧЕСТВО
    print("\nC. КАЧЕСТВО ПОИСКА НА 17 ДЕМО-ВОПРОСАХ")
    print("     (BM25 не понимает смысла: «сколько платим» он не отличает от")
    print("      «обеспечительный платёж». Именно это исправят векторы и реранкер на шаге 4.)")
    questions = {
        row["id"]: row["question"]
        for row in csv.DictReader(open(config.DATA_DIR / "demo_questions.csv", encoding="utf-8-sig"))
    }

    ranks: dict[str, int | None] = {}
    for qid, needle in GOLDEN.items():
        question = questions[qid]
        tokens = query_prep.prepare_search_tokens(question)
        results = index.search_tokens(tokens, k=50)
        ranks[qid] = next((h.rank for h in results if needle in h.chunk.body), None)

    total = len(ranks)
    for k in (1, 3, 5, 10):
        found = sum(1 for rank in ranks.values() if rank and rank <= k)
        print(f"     hit@{k:<2} = {found}/{total} = {found / total:.2f}")
    mrr = sum(1 / rank for rank in ranks.values() if rank) / total
    print(f"     MRR    = {mrr:.2f}")

    hit5 = sum(1 for rank in ranks.values() if rank and rank <= 5) / total
    hit10 = sum(1 for rank in ranks.values() if rank and rank <= 10) / total
    check(hit5 >= 0.75, f"hit@5 ≥ 0.75 (получено {hit5:.2f})")
    check(hit10 >= 0.90, f"hit@10 ≥ 0.90 (получено {hit10:.2f})")

    wrong_top1 = [qid for qid in MUST_BE_TOP1 if ranks.get(qid) != 1]
    check(not wrong_top1, "на точных совпадениях нужный фрагмент на 1-м месте",
          f"не на первом месте: {wrong_top1}")

    misses = {qid: rank for qid, rank in ranks.items() if rank is None or rank > 5}
    if misses:
        print("\n     промахи hit@5 (ожидаемо для чистого BM25):")
        for qid, rank in sorted(misses.items(), key=lambda item: item[1] or 999):
            print(f"       вопрос {qid}: место {rank} — {questions[qid][:70]}")

    # Диагностика порога отказа: сравниваем лучшую оценку на вопросе, ответа
    # на который в базе нет, с оценками на вопросах, где ответ есть.
    no_answer_score = index.max_score(QUESTION_WITHOUT_ANSWER)
    golden_scores = [index.max_score(questions[qid]) for qid in GOLDEN]
    print(f"\n     лучшая оценка на вопросе БЕЗ ответа в базе: {no_answer_score:.2f}")
    print(f"     минимальная лучшая оценка на вопросах С ответом: {min(golden_scores):.2f}")
    check(no_answer_score < max(golden_scores),
          "вопрос без ответа в базе даёт не самую высокую оценку",
          f"{no_answer_score:.2f} против {max(golden_scores):.2f}")

    print()
    if failures:
        print(f"ИТОГ: не прошло проверок — {len(failures)}. Шаг 3 не закрыт.")
        return 1
    print("ИТОГ: все проверки прошли. Шаг 3 закрыт.")
    print("Дальше: шаг 4 добавит векторы (bge-m3) и реранкер — они поднимут hit@1,")
    print("который у чистого BM25 низкий: точные совпадения он находит,")
    print("а перефразировки и близкие пункты пока путает.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
