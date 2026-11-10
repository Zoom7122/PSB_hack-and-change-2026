"""Проверки шага 7: конвейер solution.py. Без сети и без ключей.

Запуск:
    python check_step7.py
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import config
import solution
from solution import (
    FALLBACK_ANSWER, QuestionRow, build_dry_run_searcher, fallback,
    find_questions_file, load_questions, normalize_answer, solve_all,
    validate_submission, write_submission,
)

failures: list[str] = []


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


def check(condition: bool, title: str) -> None:
    print(("PASS  " if condition else "FAIL  ") + title)
    if not condition:
        failures.append(title)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="psb_step7_"))

    # ------------------------------------------------------------- A. ВХОД
    print("A. ЧТЕНИЕ ВОПРОСОВ")
    path = find_questions_file()
    check(path.name in ("questions.csv", "demo_questions.csv"),
          f"файл вопросов найден автоматически ({path.name})")
    rows = load_questions(path)
    # В демо-наборе 18 вопросов (id 1..18). В метриках поиска участвуют 17:
    # вопрос 3 («пособие по беременности») ответа в базе не имеет и служит
    # проверкой отказа, поэтому в эталонных GOLDEN его нет.
    check(len(rows) == 18, f"прочитано 18 вопросов (получено {len(rows)})")
    check(rows[0].qid == "1" and "арендная плата" in rows[0].question.lower(),
          "первый вопрос прочитан корректно (BOM не мешает)")
    check(all(row.question.strip() for row in rows), "пустых вопросов нет")

    # ------------------------------------------------------------ B. ФОРМАТ
    print("\nB. ФОРМАТ ОТВЕТА")
    long_text = " ".join(["слово"] * 200)
    trimmed = normalize_answer(long_text)
    check(len(trimmed.split()) <= solution.MAX_WORDS,
          f"длинный ответ обрезан до {solution.MAX_WORDS} слов (получено {len(trimmed.split())})")
    huge_symbols = "а" * 20_000
    check(len(normalize_answer(huge_symbols)) <= solution.MAX_CHARS,
          "слишком длинный ответ обрезан по символам")
    check(FALLBACK_ANSWER.strip() != "", "резервный ответ не пустой")
    check(len(FALLBACK_ANSWER.split()) <= 150, "резервный ответ укладывается в лимит слов")

    # --------------------------------------------------- C. ЗАПИСЬ И ПРОВЕРКА
    print("\nC. ЗАПИСЬ И ПРОВЕРКА ФАЙЛА")
    good = [solution.Solved(qid=row.qid, answer="Ответ на вопрос [1].") for row in rows]
    good_path = tmp / "submission_ok.csv"
    write_submission(good, good_path)
    check(good_path.exists(), "файл submission.csv создан")
    check(validate_submission([row.qid for row in rows], good_path) == [],
          "валидный файл проходит проверку")
    with good_path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    check(header == ["id", "answer"], f"колонки ровно id,answer ({header})")

    # испорченные файлы должны ловиться
    bad_cases = {
        "пустой ответ": [solution.Solved(qid=row.qid, answer="   " if row.qid == "2" else "ответ")
                         for row in rows],
        "длинный ответ": [solution.Solved(qid=row.qid,
                                          answer=" ".join(["слово"] * 200) if row.qid == "3" else "ответ")
                          for row in rows],
        "пропущен вопрос": [solution.Solved(qid=row.qid, answer="ответ") for row in rows if row.qid != "5"],
    }
    for name, data in bad_cases.items():
        bad_path = tmp / f"bad_{abs(hash(name))}.csv"
        # пишем в обход write_submission: он сам приводит ответ к регламенту,
        # а проверить надо именно способность validate_submission ловить брак
        with bad_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "answer"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for item in data:
                writer.writerow({"id": item.qid, "answer": item.answer})
        problems = validate_submission([row.qid for row in rows], bad_path)
        check(bool(problems), f"испорченный файл пойман: {name} ({problems[0][:60] if problems else '—'})")

    # ------------------------------------------------------ D. РЕЗЕРВНЫЙ ПУТЬ
    print("\nD. РЕЗЕРВ И БЮДЖЕТ ВРЕМЕНИ")
    tiny = [fallback(row, "тест") for row in rows[:3]]
    check(all(item.source == "fallback" for item in tiny), "резервные ответы помечены как fallback")
    with_budget_path = tmp / "submission_budget.csv"
    write_submission(tiny, with_budget_path)
    check(validate_submission([r.qid for r in rows[:3]], with_budget_path) == [],
          "файл из резервных ответов остаётся валидным")

    # ------------------------------------------------- E. КОНВЕЙЕР ЦЕЛИКОМ
    print("\nE. КОНВЕЙЕР ЦЕЛИКОМ (dry-run, без сети)")
    searcher, meta, client = build_dry_run_searcher()
    solved = solve_all(rows, searcher, client, meta, budget_sec=600, workers=2, verbose=False)
    check(len(solved) == len(rows), f"обработаны все вопросы ({len(solved)})")
    check([item.qid for item in solved] == [row.qid for row in rows],
          "порядок ответов совпадает с порядком вопросов")
    sources = {item.source for item in solved}
    check(sources <= {"model", "refused", "fallback"}, f"источники ответов корректны ({sources})")
    check(any(item.source == "refused" for item in solved),
          "вопрос вне базы получает отказ, а не выдумку")

    end_path = tmp / "submission_end.csv"
    write_submission(solved, end_path)
    check(validate_submission([row.qid for row in rows], end_path) == [],
          "итоговый файл конвейера проходит проверку формата")

    # ------------------------------------------------ F. ИСЧЕРПАНИЕ БЮДЖЕТА
    print("\nF. ЧТО БУДЕТ, ЕСЛИ ВРЕМЯ КОНЧИЛОСЬ")

    class SlowClient(solution.DryRunClient):
        def chat(self, messages, temperature=None):
            import time as _time
            _time.sleep(0.05)
            return super().chat(messages, temperature)

    slow = SlowClient()
    quick = solve_all(rows, searcher, slow, meta, budget_sec=0.01, workers=1, verbose=False)
    check(len(quick) == len(rows), "при исчерпанном бюджете возвращаются ответы на все вопросы")
    check(all(item.source == "fallback" for item in quick),
          "все ответы резервные, если время вышло сразу")
    budget_path = tmp / "submission_timeout.csv"
    write_submission(quick, budget_path)
    check(validate_submission([row.qid for row in rows], budget_path) == [],
          "файл остаётся валидным даже при полном исчерпании бюджета")

    # ------------------------------------------- G. МОДЕЛЬ УМЕРЛА ПОСРЕДИ ПРОГОНА
    print("\nG. ПРЕДОХРАНИТЕЛЬ ПРИ НЕДОСТУПНОЙ МОДЕЛИ")

    class DeadClient(solution.DryRunClient):
        def chat(self, messages, temperature=None):
            raise RuntimeError("модель недоступна")

    import time as _time
    started = _time.time()
    dead = solve_all(rows, searcher, DeadClient(), meta, budget_sec=600, workers=2, verbose=False)
    spent = _time.time() - started
    check(len(dead) == len(rows), "все вопросы получили ответ даже при мёртвой модели")
    # часть вопросов отказывает раньше вызова модели (нет ответа в базе) —
    # это тоже корректный исход; недопустим только source="model"
    check(not any(item.source == "model" for item in dead),
          f"ни один ответ не делал вид, что модель ответила ({ {i.source for i in dead} })")
    check(spent < 30, f"предохранитель не даёт тратить бюджет на ретраи ({spent:.1f} с)")
    dead_path = tmp / "submission_dead.csv"
    write_submission(dead, dead_path)
    check(validate_submission([row.qid for row in rows], dead_path) == [],
          "файл валиден даже если модель умерла целиком")

    # --------------------------------------------------------------- ИТОГ
    print()
    if failures:
        print(f"ИТОГ: не прошло проверок — {len(failures)}. Шаг 7 не закрыт.")
        return 1
    print("ИТОГ: все проверки прошли. Шаг 7 закрыт (офлайн-часть).")
    print("Боевой прогон: python solution.py   → submission.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
