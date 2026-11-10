from __future__ import annotations

import sys

import ingest
import config


def main() -> int:
    archive = ingest.find_knowledge_base(config.DATA_DIR)
    docs = ingest.load_documents_from_zip(archive)

    print(f"Архив: {archive.relative_to(config.BASE_DIR) if archive.is_relative_to(config.BASE_DIR) else archive}")
    print(ingest.summarize(docs))
    print()

    for i, doc in enumerate(docs, 1):
        first_line = next((ln for ln in doc.text.splitlines() if ln.strip()), "")
        print(f"[{i:2d}] группа={doc.group:<18} | {len(doc.text):>6} симв | {doc.doc_id}")
        print(f"     title: {doc.title}")
        print(f"     первая строка: {first_line[:90]}")
    print()

    failures: list[str] = []

    def check(ok: bool, label: str) -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}")
        if not ok:
            failures.append(label)

    print("ПРОВЕРКИ")
    check(len(docs) == 21, f"документов ровно 21 (получено {len(docs)})")

    total = sum(len(d.text) for d in docs)
    check(abs(total - 270_134) <= 1000, f"суммарно ~270 134 символа (получено {total})")

    company = [d for d in docs if d.group == "документы компании"]
    laws = [d for d in docs if d.group == "законы"]
    check(len(company) == 8, f"документов компании 8 (получено {len(company)})")
    check(len(laws) == 13, f"законов 13 (получено {len(laws)})")

    ids = [d.doc_id for d in docs]
    check(len(ids) == len(set(ids)), "все doc_id уникальны")
    check(
        all(all(ch.isascii() and (ch.isalnum() or ch in "-_") for ch in d.doc_id) for d in docs),
        "doc_id только из латиницы, цифр и дефиса",
    )

    dirty = [d.name for d in docs if any(ch in d.text for ch in ("\ufeff", "\r", "\xa0"))]
    check(not dirty, f"в текстах нет BOM, \\r и неразрывных пробелов ({len(dirty)} проблемных)")

    docs_again = ingest.load_documents_from_zip(archive)
    same = [(d.doc_id, d.text_hash) for d in docs] == [(d.doc_id, d.text_hash) for d in docs_again]
    check(same, "повторное чтение даёт те же doc_id и text_hash")

    print()
    if failures:
        print(f"ИТОГ: {len(failures)} проверок не прошло — шаг 1 не закрыт.")
        return 1
    print("ИТОГ: все проверки прошли. Шаг 1 закрыт.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
