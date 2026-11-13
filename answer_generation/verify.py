"""Повторная проверка и исправление сохранённых ответов и источников."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from pydantic import ValidationError

from query_processing_module.query_processing.config import LLMSettings
from query_processing_module.query_processing.llm_client import OpenAIClient

from .models import Claim
from .service import INSUFFICIENT, check_claims, generate_verified, max_retries
from rag_pipeline.corpus import Corpus
from rag_pipeline.indexing import artifacts_dir, create_search, data_dir
from rag_pipeline.output import write_submission, write_trace
from rag_pipeline.retrieval import retrieve


def _read_trace(path: Path) -> list[dict]:
    """Читает журнал ответов и проверяет уникальность идентификаторов."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("В журнале повторяются идентификаторы вопросов")
    return records


def _read_answers(path: Path) -> dict[str, str]:
    """Читает проверяемый CSV и отклоняет неоднозначные идентификаторы."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"id", "answer"}.issubset(reader.fieldnames):
            raise ValueError("В CSV нужны колонки id и answer")
        rows = list(reader)
    result = {row["id"]: row["answer"] for row in rows}
    if len(result) != len(rows):
        raise ValueError("В CSV повторяются идентификаторы вопросов")
    return result


async def _existing_errors(client, record: dict, answer: str, corpus: Corpus) -> list[str]:
    """Проверяет, что сохранённый ответ совпадает с журналом и его источниками."""
    if record.get("answer") != answer:
        return ["Ответ в CSV отличается от журнала"]
    if record.get("status") != "answered":
        return []
    if record.get("corpus_version") != corpus.version:
        return ["Версия базы знаний изменилась"]
    try:
        claims = [Claim.model_validate(item) for item in record.get("claims", [])]
        if not claims:
            return ["У ответа нет утверждений со ссылками"]
        request = record.get("search_request") or {}
        event_date = (request.get("filters") or {}).get("event_date")
        rendered, errors = await check_claims(
            client, record.get("resolved_question", record["question"]), claims,
            record.get("sources", []), corpus, event_date
        )
        if not errors and rendered != answer:
            errors.append("Ответ не совпадает с проверенными утверждениями и ссылками")
        return errors
    except (ValueError, ValidationError, KeyError, TypeError) as exc:
        return [f"Неверный формат журнала или проверки: {type(exc).__name__}"]
    except Exception as exc:
        return [f"Проверка недоступна: {type(exc).__name__}"]


def _trusted_sources(record: dict, corpus: Corpus) -> list[dict]:
    """Восстанавливает контекст только из фрагментов актуального корпуса."""
    if record.get("corpus_version") != corpus.version:
        return []
    ids = [source.get("chunk_id") for source in record.get("sources", [])]
    return [dict(corpus.sources[cid]) for cid in dict.fromkeys(ids) if cid in corpus.sources]


async def _repair_record(client, record: dict, corpus: Corpus, team, retries: int) -> dict:
    """Повторно генерирует ответ по доверенным или заново найденным фрагментам."""
    request = record.get("search_request") or {}
    sources = _trusted_sources(record, corpus)
    if not sources and request:
        question = record.get("resolved_question") or request.get("standalone_query") or record["question"]
        sources = await asyncio.to_thread(retrieve, corpus, team, question, request)
    event_date = (request.get("filters") or {}).get("event_date")
    result = await generate_verified(client, record.get("resolved_question", record["question"]), sources, corpus,
                                     event_date=event_date, retries=retries)
    return {**record, **result, "corpus_version": corpus.version}


async def verify_records(records: list[dict], answers: dict[str, str], corpus: Corpus,
                         client, team, retries: int) -> list[dict]:
    """Проверяет каждую строку и исправляет ответы с ошибками в источниках."""
    if set(answers) != {record["id"] for record in records}:
        raise ValueError("Идентификаторы CSV и журнала не совпадают")
    output = []
    for record in records:
        errors = await _existing_errors(client, record, answers[record["id"]], corpus)
        if not errors:
            status = "passed" if record.get("status") == "answered" else "not_applicable"
            output.append({**record, "reverification": {"status": status, "errors": []}})
            continue
        try:
            repaired = await _repair_record(client, record, corpus, team, retries)
        except Exception as exc:
            repaired = {**record, "status": "insufficient", "answer": INSUFFICIENT,
                        "claims": [], "sources": [], "corpus_version": corpus.version,
                        "verification": {"status": "not_passed", "errors": [type(exc).__name__]}}
        repaired["recheck_errors"] = errors
        repaired["reverification"] = {"status": repaired["verification"]["status"], "errors": errors}
        output.append(repaired)
    return output


def main() -> None:
    """Запускает проверку CSV и записывает исправленные копии результатов."""
    parser = argparse.ArgumentParser(description="Проверить и исправить ссылки в ответах")
    parser.add_argument("--trace", type=Path, default=artifacts_dir() / "answer_trace.jsonl")
    parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    parser.add_argument("--out", type=Path, default=Path("submission.verified.csv"))
    parser.add_argument("--out-trace", type=Path, default=artifacts_dir() / "answer_trace.verified.jsonl")
    args = parser.parse_args()
    records, answers = _read_trace(args.trace), _read_answers(args.submission)
    settings = LLMSettings.from_environment()
    corpus = Corpus.from_directory(data_dir())
    team = create_search(corpus, artifacts_dir())

    async def run() -> list[dict]:
        """Создаёт общий клиент и проверяет сохранённые ответы."""
        client = OpenAIClient(settings)
        try:
            return await verify_records(records, answers, corpus, client, team, max_retries())
        finally:
            await client.close()

    try:
        verified = asyncio.run(run())
    finally:
        team.close()
    write_trace(verified, args.out_trace)
    write_submission(verified, args.out)
    print(f"Проверено ответов: {len(verified)}; результат: {args.out}")


if __name__ == "__main__":
    main()
