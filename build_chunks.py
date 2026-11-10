from __future__ import annotations

import argparse
import json
from pathlib import Path

import chunking
import config
import ingest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=0, help="показать N примеров фрагментов")
    parser.add_argument("--max-chars", type=int, default=1400, help="максимальная длина тела фрагмента")
    args = parser.parse_args()

    archive = ingest.find_knowledge_base(config.DATA_DIR)
    docs = ingest.load_documents_from_zip(archive)
    print(f"База знаний: {archive}")
    print(ingest.summarize(docs))
    print()

    chunks, metas = chunking.build_chunks(docs, max_chars=args.max_chars)
    print()

    # --- статистика: её смотрим раньше, чем примеры --------------------------
    sizes = sorted(len(c.body) for c in chunks)
    n = len(sizes)
    print(f"Всего фрагментов: {n}")
    print(f"Длина тела: min {sizes[0]}, медиана {sizes[n // 2]}, "
          f"p90 {sizes[int(n * 0.9)]}, max {sizes[-1]} символов")
    print(f"Коротких (<60 симв.): {sum(1 for c in chunks if len(c.body) < 60)}")
    print(f"Фрагментов с вклеенной инструкцией: {sum(1 for c in chunks if c.has_injection)}")
    void_docs = [m for m in metas.values() if m.status != "действует"]
    print(f"Документов «утратил силу»: {len(void_docs)}")
    for meta in void_docs:
        print(f"  - {meta.citation_title}")

    # --- сохранение ----------------------------------------------------------
    out = Path(config.BUILD_DIR)
    out.mkdir(parents=True, exist_ok=True)

    chunks_path = out / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    docs_path = out / "docs.json"
    docs_path.write_text(
        json.dumps({k: v.__dict__ for k, v in metas.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nСохранено: {chunks_path} ({n} строк)")
    print(f"Сохранено: {docs_path}")

    # --- примеры для проверки глазами ---------------------------------------
    if args.show:
        print("\n" + "=" * 100)
        print("ПРИМЕРЫ ФРАГМЕНТОВ — проверьте: пункт не разорван, шапка на месте")
        print("=" * 100)
        step = max(1, n // args.show)
        for chunk in chunks[::step][: args.show]:
            print(f"\n--- {chunk.chunk_id}")
            print(chunk.text[:600])


if __name__ == "__main__":
    main()
