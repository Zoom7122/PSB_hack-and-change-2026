"""Офлайн-проверка покрытия фрагментов из демонстрационных ответов, без LLM."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_processing.hybrid import HybridSearcher
from data_processing.search import build_index
from data_processing.vector_index import VectorIndex
from rag_pipeline.corpus import Corpus
from rag_pipeline.indexing import TeamSearch
from rag_pipeline.retrieval import retrieve


def matches(requirement: dict, source: dict) -> bool:
    """Проверяет наличие нужного документа и положения из ручной разметки эталона."""
    return (
        requirement["document"].casefold() in source["document_title"].casefold()
        and (not requirement.get("section") or requirement["section"] == source["section"])
        and (not requirement.get("text") or requirement["text"].casefold() in source["text"].casefold())
    )


def coverage(records: list[dict], requirements: dict) -> dict:
    """Считает покрытие доказательств; это не оценка юридического ответа LLM."""
    found, total, complete, answerable, details = 0, 0, 0, 0, []
    by_id = {record["id"]: record for record in records}
    for row_id, expected in requirements.items():
        if not expected:
            continue
        sources = by_id.get(row_id, {}).get("sources", [])
        flags = [any(matches(item, source) for source in sources) for item in expected]
        found += sum(flags)
        total += len(flags)
        answerable += 1
        complete += all(flags)
        details.append({"id": row_id, "found": sum(flags), "required": len(flags),
                        "source_count": len(sources),
                        "missing": [item for item, present in zip(expected, flags) if not present]})
    return {"evidence_found": found, "evidence_total": total, "evidence_recall": found / total,
            "questions_fully_covered": complete, "answerable_questions": answerable, "questions": details}


def main() -> None:
    """Сравнивает прежнюю трассу и текущий поиск на одинаковых исходных вопросах."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data/demo_questions.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--evidence", type=Path, default=ROOT / "tests/fixtures/demo_evidence.json")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/demo_retrieval_report.json")
    args = parser.parse_args()
    requirements = json.loads(args.evidence.read_text(encoding="utf-8"))
    corpus = Corpus.from_directory(args.data_dir)
    lexical = build_index(corpus.chunks)
    vectors = VectorIndex(corpus.chunks, np.zeros((len(corpus.chunks), 1), dtype=np.float32))
    team = TeamSearch(HybridSearcher(lexical, vectors), None)
    with args.questions.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    records, top_six = [], []
    for row in rows:
        request = {"standalone_query": row["question"], "search_queries": [], "filters": {}}
        records.append({**row, "sources": retrieve(corpus, team, row["question"], request)})
        top_six.append({**row, "sources": retrieve(corpus, team, row["question"], request, limit=6)})
    report = {"mode": "BM25 only; no LLM; not a platform score",
              "current_top6": coverage(top_six, requirements),
              "current_context": coverage(records, requirements)}
    if args.baseline:
        report["baseline_context"] = coverage(json.loads(args.baseline.read_text(encoding="utf-8")), requirements)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.out.with_name("demo_retrieval_trace.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, result in report.items():
        if isinstance(result, dict):
            print(name, result["evidence_found"], "/", result["evidence_total"],
                  "fully covered:", result["questions_fully_covered"], "/", result["answerable_questions"])
    print("Report:", args.out)


if __name__ == "__main__":
    main()
