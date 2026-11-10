"""Самопроверка шага 2: метаданные и нарезка на фрагменты.

Запуск:
    python3 check_step2.py
    python3 check_step2.py --show 5     # плюс показать 5 фрагментов

Проверяем три вещи:
  A. СТРУКТУРА — фрагменты не обрезаны, не дублируются, шапка на месте.
  B. МЕТАДАННЫЕ — типы документов, редакции, статусы, контрагенты, ИНН.
  C. ФАКТЫ — в нужных пунктах лежат те самые цифры и формулировки, которые
     спросят демо-вопросы. Это главная часть: даже идеальная структура
     бесполезна, если пункт 3.1 не содержит «450 000».
"""

from __future__ import annotations

import argparse
import sys

import chunking
import config
import ingest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0)
    args = parser.parse_args()

    docs = ingest.load_documents_from_zip(ingest.find_knowledge_base(config.DATA_DIR))
    chunks, metas = chunking.build_chunks(docs, verbose=False)
    print(f"Документов: {len(docs)}, фрагментов: {len(chunks)}")
    print()

    failures: list[str] = []

    def check(ok: bool, label: str, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {label}" + (f"  → {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    def chunk_with(text: str) -> chunking.Chunk | None:
        return next((c for c in chunks if text in c.body), None)

    # ---------------------------------------------------------------- A. СТРУКТУРА
    print("A. СТРУКТУРА")
    check(300 <= len(chunks) <= 700, f"фрагментов 300–700 (получено {len(chunks)})")
    check(all(len(c.body) <= 1400 for c in chunks), "ни один фрагмент не длиннее 1400 символов")

    tiny = [c for c in chunks if len(c.body) < 40]
    check(not tiny, f"нет фрагментов короче 40 символов (найдено {len(tiny)})",
          "; ".join(f"{c.chunk_id}={c.body!r}" for c in tiny[:3]))

    ids = [c.chunk_id for c in chunks]
    check(len(ids) == len(set(ids)), "chunk_id уникальны")

    no_prefix = [c for c in chunks if not c.prefix.strip() or not c.label.strip()]
    check(not no_prefix, f"у каждого фрагмента есть шапка и подпись (проблемных {len(no_prefix)})")

    duplicate_numbering = [c for c in chunks if c.body.strip()[:12].count(". ") >= 2
                           and c.body.strip()[:12].split(". ")[0] == c.body.strip()[:12].split(". ")[1]]
    check(not duplicate_numbering, "нет задвоенных номеров вида «5.3. 5.3. …»")

    # Пункты договоров должны быть целиком, а не «первые N символов документа»
    arenda = next(m for m in metas.values() if m.number == "А-08/2024")
    arenda_chunks = [c for c in chunks if c.doc_id == arenda.doc_id]
    clause_count = sum(1 for line in docs[0].text.split("\n")
                       if chunking.RE_CLAUSE.match(line.strip())) if docs else 0
    contract_doc = next(d for d in docs if d.doc_id == arenda.doc_id)
    expected = sum(1 for line in contract_doc.text.split("\n") if chunking.RE_CLAUSE.match(line.strip()))
    got = sum(1 for c in arenda_chunks if c.label.startswith("п. "))
    check(got == expected, f"все пункты договора аренды нарезаны (ожидалось {expected}, получено {got})")

    # ------------------------------------------------------------- B. МЕТАДАННЫЕ
    print("\nB. МЕТАДАННЫЕ")
    kinds = {}
    for meta in metas.values():
        kinds[meta.kind] = kinds.get(meta.kind, 0) + 1
    print(f"     типы документов: {kinds}")
    check(kinds.get("закон") == 13, f"13 законов (получено {kinds.get('закон')})")
    check(kinds.get("договор") == 3, f"3 договора (получено {kinds.get('договор')})")
    check(kinds.get("доп.соглашение") == 1, "доп. соглашение распознано")
    check(kinds.get("регламент") == 2, f"2 регламента (получено {kinds.get('регламент')})")
    check(kinds.get("письмо") == 1, "письмо комплаенс распознано")
    check(kinds.get("записка") == 1, "служебная записка распознана")

    # Статус сверяем по дате редакции, а не по порядку в списке: строки
    # «01.03.2025» и «10.01.2023» при обычной сортировке идут не по времени.
    status_by_edition = {m.edition: m.status for m in metas.values() if m.kind == "регламент"}
    check(status_by_edition.get("10.01.2023") == "утратил силу"
          and status_by_edition.get("01.03.2025") == "действует",
          "регламент ред. 10.01.2023 — утратил силу, ред. 01.03.2025 — действует",
          str(status_by_edition))

    check(all(m.status == "действует" for m in metas.values() if m.kind == "закон"),
          "все законы помечены как действующие (в базе нет старых редакций)")

    dop = next((m for m in metas.values() if m.kind == "доп.соглашение"), None)
    check(dop is not None and dop.parent_contract == "П-17/2024",
          f"у доп. соглашения заполнен материнский договор (получено {dop.parent_contract if dop else '—'})")

    contracts = {m.number: m for m in metas.values() if m.kind == "договор"}
    check(set(contracts) == {"А-08/2024", "П-17/2024", "ПТ-05/2023"},
          f"номера договоров распознаны: {sorted(contracts)}")
    check(contracts.get("П-17/2024") and contracts["П-17/2024"].inn != contracts["ПТ-05/2023"].inn,
          "«Альфа-Трейд» и «Альфа-Трейдинг» различаются по ИНН")

    citation_expectations = {
        "А-08/2024": "Договор аренды № А-08/2024",
        "ПТ-05/2023": "Договор поставки № ПТ-05/2023",
    }
    for number, prefix in citation_expectations.items():
        meta = contracts[number]
        check(meta.citation_title.startswith(prefix),
              f"ссылка на {number} выглядит как «{prefix}…» (получено «{meta.citation_title[:45]}»)")

    law_titles = {m.citation_title for m in metas.values() if m.kind == "закон"}
    check({"ТК РФ, ст. 70", "НК РФ, ст. 164", "152-ФЗ, ст. 9"} <= law_titles,
          "ссылки на законы в формате «ТК РФ, ст. 70»")

    injections = [c for c in chunks if c.has_injection]
    check(len(injections) == 1, f"ровно 1 фрагмент с вклеенной инструкцией (найдено {len(injections)})")
    check(bool(injections) and "записка" in next(
        m.citation_title.lower() for m in metas.values() if m.doc_id == injections[0].doc_id),
        "инъекция найдена именно в служебной записке")

    # ------------------------------------------------------------------ C. ФАКТЫ
    print("\nC. ФАКТЫ В НУЖНЫХ ПУНКТАХ")
    facts = [
        ("450 000", "п. 3.1", "арендная плата 450 000"),
        ("не позднее 5-го числа", "п. 3.2", "срок внесения платы"),
        ("не более чем на 7%", "п. 3.4", "предел индексации"),
        ("круглосуточно", "п. 4.3", "доступ транспорта 24/7"),
        ("по акту в состоянии", "п. 6.3", "возврат помещения по акту"),
        ("акта о выявленных недостатках", "п. 6.2", "акт к претензии"),
        ("5 (пяти) рабочих дней с даты поставки", "п. 6.1", "приёмка по качеству 5 р.д."),
        ("24 (двадцать четыре) месяца", "п. 5.2", "гарантия П-17 24 месяца"),
        ("включает НДС по ставке 20%, стоимость упаковки и доставки", "п. 2.1", "доставка в цене"),
        ("гарантийном талоне", "п. 6.1", "гарантия ПТ-05 по талону"),
        ("допускается предоплата", "п. 4.2", "предоплата ПТ-05"),
        ("не более 3 (трёх) рабочих дней", None, "выдача оригиналов 3 р.д."),
    ]
    for needle, expected_label, human in facts:
        chunk = chunk_with(needle)
        ok = chunk is not None and (expected_label is None or chunk.label.startswith(expected_label))
        check(ok, f"{human}: «{needle[:40]}» в {expected_label or 'любом фрагменте'}",
              f"найдено в {chunk.label if chunk else 'нигде'}")

    # Пороговые суммы регламентов — их нельзя перепутать между редакциями
    reg2023 = next(m for m in metas.values() if m.kind == "регламент" and m.status != "действует")
    reg2025 = next(m for m in metas.values() if m.kind == "регламент" and m.status == "действует")
    c2023 = next((c for c in chunks if c.doc_id == reg2023.doc_id and "10 000 000" in c.body), None)
    c2025 = next((c for c in chunks if c.doc_id == reg2025.doc_id and "15 000 000" in c.body), None)
    check(c2023 is not None and c2025 is not None, "пороги 10 млн (ред. 2023) и 15 млн (ред. 2025) на месте")
    check(c2023 is not None and "УТРАТИЛА СИЛУ" in c2023.prefix,
          "во фрагменте утратившей силу редакции есть пометка в шапке")

    # В шапке закона не должно быть служебных строк файла
    dirty_prefix = [c for c in chunks if "base.garant.ru" in c.prefix or "Дата выгрузки" in c.prefix]
    check(not dirty_prefix, f"служебные строки (URL, дата выгрузки) не попали в шапку ({len(dirty_prefix)})")

    # ---------------------------------------------------------------- ИТОГ
    print()
    if args.show:
        print("=" * 100)
        step = max(1, len(chunks) // args.show)
        for chunk in chunks[::step][: args.show]:
            print(f"\n--- {chunk.chunk_id}\n{chunk.text[:500]}")

    if failures:
        print(f"ИТОГ: не прошло проверок — {len(failures)}. Шаг 2 не закрыт.")
        return 1
    print("ИТОГ: все проверки прошли. Шаг 2 закрыт.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
