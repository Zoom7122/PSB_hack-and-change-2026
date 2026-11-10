"""Проверки шага 5: разбор вопроса. Без сети и без ключей.

Запуск:
    python check_step5.py
"""

from __future__ import annotations

import csv
import sys
from datetime import date

import config
import question_analysis as qa

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
    meta = qa.load_doc_meta()
    check(bool(meta), f"метаданные базы загружены ({len(meta)} документов)")
    if not meta:
        print("\n  Метаданные не найдены. Что делать:")
        print("    1. python build_chunks.py   — создаст artifacts/docs.json")
        print("    2. проверить, что data/knowledge-base.zip на месте")
        print("  Без метаданных не работают: контрагенты, редакции, статусы —"
              "\n  то есть блоки C и E ниже упадут неизбежно.")
        print("\n  Остальные проверки (суммы, реквизиты, даты) работают и без них.\n")

    questions = {
        row["id"]: row["question"]
        for row in csv.DictReader(open(config.DATA_DIR / "demo_questions.csv", encoding="utf-8-sig"))
    }

    # ---------------------------------------------------------------- A. суммы
    print("\nA. ИЗВЛЕЧЕНИЕ СУММ")
    check(qa.parse_amounts("арендная плата 450 000 рублей") == [450_000],
          "«450 000 рублей» → 450 000")
    check(qa.parse_amounts("договор на 12 млн рублей") == [12_000_000],
          "«12 млн рублей» → 12 000 000")
    check(qa.parse_amounts("1 200 000,00") == [1_200_000], "«1 200 000,00» → 1 200 000")
    check(qa.parse_amounts("в течение 3 рабочих дней") == [],
          "«3 рабочих дней» не считается суммой (меньше 1000)")
    check(qa.parse_amounts("заключали в августе 2024 года") == [],
          "год 2024 не считается суммой")
    check(qa.parse_amounts("2 024 рубля") == [2024], "«2 024 рубля» с разбивкой — всё-таки сумма")

    # ------------------------------------------------------------- B. реквизиты
    print("\nB. РЕКВИЗИТЫ")
    check(qa.parse_inns("ИНН 7701234567 у контрагента") == ["7701234567"], "ИНН из 10 цифр")
    check(qa.parse_contract_numbers("по договору № А-08/2024") == ["А-08/2024"],
          "номер договора А-08/2024")
    check(qa.parse_contract_numbers("ПТ-05/2023 и П-17/2024") == ["П-17/2024", "ПТ-05/2023"],
          "два номера договоров")
    check(qa.parse_dates("заключали в августе 2024 года") == [date(2024, 8, 1)],
          "«в августе 2024» → 01.08.2024")
    check(qa.parse_dates("с 01.08.2024") == [date(2024, 8, 1)], "дата 01.08.2024")

    # ------------------------------------------------------- C. контрагенты
    print("\nC. РАЗЛИЧЕНИЕ ПОХОЖИХ КОНТРАГЕНТОВ")
    trade = qa.find_counterparties("поставка для ООО «Альфа-Трейд»", meta)
    trading = qa.find_counterparties("поставка для ООО «Альфа-Трейдинг»", meta)
    check(trade == ["Альфа-Трейд"], f"«Альфа-Трейд» распознан отдельно ({trade})")
    check(trading == ["Альфа-Трейдинг"], f"«Альфа-Трейдинг» распознан отдельно ({trading})")
    check(set(trade).isdisjoint(trading), "похожие контрагенты не смешиваются")
    check(qa.find_counterparties("по договору с ООО «Северный Склад»", meta) == ["Северный Склад"],
          "«Северный Склад» распознан")

    # ------------------------------------------------- D. ложные посылки
    print("\nD. ЛОЖНЫЕ ПОСЫЛКИ")
    a4 = qa.analyze(questions["4"], meta)
    check(a4.kind == "false_premise", f"вопрос 4 → ложная посылка (получено {a4.kind})")
    check(any(c.value == "500 000" for c in a4.claims),
          f"сумма 500 000 помечена как утверждение для проверки ({[str(c) for c in a4.claims]})")
    a1 = qa.analyze(questions["1"], meta)
    check(a1.kind == "factoid", f"вопрос 1 без ложной посылки → factoid (получено {a1.kind})")
    check(not a1.claims, "в обычном вопросе утверждений для проверки нет")

    # ------------------------------------------------- E. редакции по дате
    print("\nE. ВЫБОР РЕДАКЦИИ ПО ДАТЕ СОБЫТИЯ")
    a5 = qa.analyze(questions["5"], meta)
    check(a5.kind == "edition_dependent", f"вопрос 5 → зависит от редакции (получено {a5.kind})")
    check(a5.event_date == date(2024, 8, 1), f"дата события извлечена ({a5.event_date})")
    check(a5.preferred_doc_ids == ["reglament-soglasovaniya-dogovorov-red-2023-utrat-ac1cc7"],
          f"выбрана редакция 2023 года, действовавшая на дату события ({a5.preferred_doc_ids})")
    check("reglament-soglasovaniya-dogovorov-red-2025-2c48c3" in a5.superseded_doc_ids,
          "редакция 2025 года помечена, а не выброшена")
    check(any("2024" in note for note in a5.notes), "пояснение о применимой редакции есть")

    # без даты события — берём действующую редакцию
    a10 = qa.analyze(questions["10"], meta)
    check("reglament-soglasovaniya-dogovorov-red-2025-2c48c3" in a10.preferred_doc_ids,
          "без даты события выбрана действующая редакция")

    # ----------------------------------------------------- F. типы вопросов
    print("\nF. ТИПЫ ВОПРОСОВ НА ДЕМО-НАБОРЕ")
    kinds = {qid: qa.analyze(text, meta).kind for qid, text in questions.items()}
    for qid in sorted(kinds, key=int):
        print(f"      вопрос {qid:>2} → {kinds[qid]}")
    check(kinds["3"] == "out_of_scope", "вопрос 3 (пособие по беременности) → вне базы")
    check(kinds["10"] == "multi_source", "вопрос 10 (кто запускает согласование) → несколько источников")
    check(qa.analyze(questions["15"], meta).is_dialog, "вопрос 15 распознан как диалог")
    check(sum(1 for kind in kinds.values() if kind == "out_of_scope") == 1,
          "вне базы только один вопрос из демо-набора")

    # --------------------------------------------------------------- ИТОГ
    print()
    if failures:
        print(f"ИТОГ: не прошло проверок — {len(failures)}. Шаг 5 не закрыт.")
        return 1
    print("ИТОГ: все проверки прошли. Шаг 5 закрыт.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
