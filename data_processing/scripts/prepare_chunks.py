"""Собрать фрагменты базы знаний в `artifacts/chunks.jsonl` — без сети.

Зачем
=====
Модулям 3 и 4 нужны фрагменты, чтобы их можно было проверить. Полный конвейер
(`solution.py`) строит ещё и векторный индекс, а для этого нужны эмбеддинги и
ключ провайдера. Для проверки контракта векторы не нужны: BM25-поиска
достаточно, чтобы получить реальных кандидатов и реальные метаданные.

Поэтому здесь только два шага: ZIP с документами → извлечение текста →
нарезка на фрагменты → JSONL. Ни одного сетевого вызова, работает офлайн.

Запуск
======
    python scripts/prepare_chunks.py
    python scripts/prepare_chunks.py --zip data/knowledge-base.zip --out artifacts/chunks.jsonl

Файл `artifacts/chunks.jsonl` в git не коммитится: он собирается за секунды
из `data/knowledge-base.zip`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE = Path(__file__).resolve().parent.parent      # корень проекта
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import chunking  # noqa: E402
import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Собрать фрагменты базы без сети")
    parser.add_argument("--zip", default=str(BASE / "data" / "knowledge-base.zip"),
                        help="ZIP с документами базы знаний")
    parser.add_argument("--out", default=str(BASE / "artifacts" / "chunks.jsonl"),
                        help="куда записать JSONL")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_file():
        raise SystemExit(f"Нет файла {zip_path}. Положите базу знаний в data/")

    documents = ingest.load_documents_from_zip(zip_path)
    chunks, metas = chunking.build_chunks(documents, verbose=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    kinds: dict[str, int] = {}
    for meta in metas.values():
        kinds[meta.kind] = kinds.get(meta.kind, 0) + 1

    print(f"документов: {len(documents)} | фрагментов: {len(chunks)}")
    print("по типам документов: " + ", ".join(f"{k} — {v}" for k, v in sorted(kinds.items())))
    print(f"записано: {out}")
    print("дальше: python tests/test_contract.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
