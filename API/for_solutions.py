"""Платформенный CSV-пайплайн подготовки, поиска и проверки ответов."""

import asyncio
import argparse
import csv
import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from query_processing_module.query_processing.config import LLMSettings
from query_processing_module.query_processing.contracts import QueryRequest
from query_processing_module.query_processing.llm_client import LLMClient, OpenAIClient
from query_processing_module.query_processing.multi_stage import MultiStageQueryProcessingService
from query_processing_module.query_processing.service import QueryProcessingService

from API.backend_api import prepare_query
from answer_generation.service import INSUFFICIENT, generate_verified, max_retries
from rag_pipeline.corpus import Corpus
from rag_pipeline.indexing import artifacts_dir, create_search, data_dir
from rag_pipeline.output import write_submission, write_trace
from rag_pipeline.retrieval import retrieve
from data_processing.contract_adapter import normalize_number

LOGGER = logging.getLogger("solution")
SERVICE_TYPES = {"single": QueryProcessingService, "multi": MultiStageQueryProcessingService}


class QueryService(Protocol):
    """Описывает общий вход сервисов single- и multi-stage режимов."""

    async def process_query(self, request: dict | QueryRequest) -> dict:
        """Обрабатывает один валидированный запрос."""
        ...


@dataclass(frozen=True)
class QuestionRow:
    """Хранит исходный ID, вопрос и проверенный запрос одной CSV-строки."""

    row_id: str
    request: QueryRequest
    original_question: str


class LoggingLLMClient:
    """Логирует ответы модели с маскированием точного значения API-ключа."""

    def __init__(self, client: LLMClient, row_id: str, api_key: str) -> None:
        """Связывает общий LLM-клиент с ID текущего вопроса."""
        self._client = client
        self._row_id = row_id
        self._api_key = api_key
        self._call_number = 0

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Запрашивает модель и пишет её ответ в JSON-лог."""
        response = await self._client.complete(messages)
        self._call_number += 1
        log = LOGGER.info if os.getenv("LLM_LOG_RESPONSES") == "1" else LOGGER.debug
        log("LLM response %s", json.dumps({
            "id": self._row_id, "call": self._call_number,
            "response": _redact_secret(response, self._api_key),
        }, ensure_ascii=False))
        return response


def _redact_secret(value: str, secret: str) -> str:
    """Заменяет точные вхождения API-ключа перед записью в лог."""
    return value.replace(secret, "[REDACTED_SECRET]") if secret else value


def load_questions(directory: Path, path: Path | None = None) -> list[QuestionRow]:
    """Читает CSV, проверяет ID и формирует запросы по контракту."""
    with (path or directory / "questions.csv").open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"id", "question"}.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")
        rows, seen_ids = [], set()
        for line_number, row in enumerate(reader, start=2):
            row_id, question = row.get("id"), row.get("question")
            if row_id is None or not row_id.strip():
                raise ValueError(f"В questions.csv пустой id в строке {line_number}")
            if row_id in seen_ids:
                raise ValueError(f"Повторяющийся id в questions.csv: {row_id}")
            if question is None:
                raise ValueError(f"Отсутствует question в строке {line_number}")
            seen_ids.add(row_id)
            rows.append(QuestionRow(row_id, prepare_query(question), question.strip()))
    if not rows:
        raise ValueError("В questions.csv нет вопросов")
    return rows


def _service_type(mode: str) -> type[QueryService]:
    """Выбирает реализацию Query Processing по переменной режима."""
    try:
        return SERVICE_TYPES[mode.strip().lower()]
    except KeyError:
        raise ValueError("QUERY_PROCESSING_MODE must be single or multi") from None


def _worker_count() -> int:
    """Читает MAX_WORKERS и ограничивает параллельность диапазоном 1–8."""
    try:
        return min(8, max(1, int(os.getenv("MAX_WORKERS", "2"))))
    except ValueError:
        return 2


def _early_result(status: str, message: str, reason: str = "") -> dict:
    """Создаёт запись для уточнения, отказа или сбоя обработки."""
    return {
        "status": status, "answer": message, "claims": [], "sources": [],
        "attempts": [], "verification": {"status": "not_applicable", "errors": [reason] if reason else []},
    }


async def _prepare_search(row: QuestionRow, client, service_class) -> tuple[dict, str | None]:
    """При сбое подготовки сохраняет возможность поиска по репликам пользователя."""
    try:
        return await service_class(client).process_query(row.request), None
    except Exception as exc:
        context = " ".join(message.content for message in row.request.history if message.role == "user")
        standalone = f"{context}\n{row.request.question}".strip()
        return {"status": "ready", "search_request": {
            "standalone_query": standalone, "search_queries": [row.request.question],
            "filters": {"document_number": None, "document_type": None, "event_date": None},
        }}, type(exc).__name__


def _ground_filters(request: dict, original: str) -> dict:
    """Убирает номера, придуманные при подготовке, вместо обнуления выдачи."""
    result = dict(request)
    filters = dict(result.get("filters") or {})
    number = normalize_number(filters.get("document_number"))
    if number and (number not in (normalize_number(original) or "") or filters.get("document_type") in {"law", "закон"}):
        filters["document_number"] = None
    result["filters"] = filters
    return result


async def _run_row(row: QuestionRow, client: LLMClient, service_class: type[QueryService],
                   corpus: Corpus, team, retries: int) -> dict:
    """Обрабатывает один вопрос от Query Processing до проверенного ответа."""
    base = {
        "id": row.row_id, "question": row.original_question,
        "corpus_version": corpus.version, "search_request": None,
    }
    started = time.monotonic()
    try:
        query_result, issue = await _prepare_search(row, client, service_class)
        base["query_processing_error"] = issue
        status = query_result["status"]
        if status != "ready":
            result = _early_result(status, query_result["message"])
        else:
            request = _ground_filters(query_result["search_request"], row.original_question)
            base["search_request"] = request
            resolved = request["standalone_query"]
            base["resolved_question"] = resolved
            sources = await asyncio.to_thread(retrieve, corpus, team, resolved, request)
            result = await generate_verified(
                client, resolved, sources, corpus,
                event_date=request.get("filters", {}).get("event_date"), retries=retries,
            )
    except Exception as exc:
        LOGGER.error("Ошибка запроса id=%s type=%s", row.row_id, type(exc).__name__)
        result = _early_result("insufficient", INSUFFICIENT, type(exc).__name__)
    base["diagnostics"] = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "source_count": len(result.get("sources", [])),
        "document_count": len({item["document_id"] for item in result.get("sources", [])}),
        "llm_calls": getattr(client, "_call_number", None),
        "answer_words": len(result["answer"].split()),
    }
    LOGGER.info("Результат %s", json.dumps({"id": row.row_id, "status": result["status"],
                **base["diagnostics"], "errors": result["verification"]["errors"][:3]}, ensure_ascii=False))
    return {**base, **result}


async def _process_rows(rows: list[QuestionRow], settings: LLMSettings,
                        service_class: type[QueryService], corpus: Corpus,
                        team, retries: int, *, checkpoint=None,
                        time_budget: float | None = None) -> list[dict]:
    """Сохраняет завершённые ответы и прекращает запросы до лимита платформы."""
    client = OpenAIClient(settings)
    if time_budget is not None and getattr(team, "client", None) is not None:
        team.client.deadline = time.monotonic() + time_budget
    semaphore = asyncio.Semaphore(_worker_count())

    records = [{"id": row.row_id, "question": row.original_question, "corpus_version": corpus.version,
                **_early_result("insufficient", INSUFFICIENT, "not_processed")} for row in rows]
    if checkpoint:
        checkpoint(records)

    async def process(index: int, row: QuestionRow) -> None:
        """Ограничивает число одновременно обрабатываемых вопросов."""
        async with semaphore:
            wrapped = LoggingLLMClient(client, row.row_id, settings.api_key)
            records[index] = await _run_row(row, wrapped, service_class, corpus, team, retries)
            if checkpoint:
                checkpoint(records)

    try:
        work = asyncio.gather(*(process(index, row) for index, row in enumerate(rows)))
        try:
            await asyncio.wait_for(work, timeout=time_budget)
        except asyncio.TimeoutError:
            LOGGER.error("Исчерпан бюджет времени; завершённые ответы сохранены")
            for record in records:
                if record["verification"]["errors"] == ["not_processed"]:
                    record["verification"]["errors"] = ["time_budget_exhausted"]
        return records
    finally:
        await client.close()


def _arguments(argv: list[str] | None):
    """Разрешает проверять демо в отдельных файлах без изменения платформенных входов."""
    parser = argparse.ArgumentParser(description="Юридический RAG: вопросы CSV -> проверенные ответы CSV")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--mode", choices=tuple(SERVICE_TYPES))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv or [])
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit должен быть положительным")
    return args


def main(argv: list[str] | None = None) -> None:
    """Запускает поиск и генерацию, затем сохраняет ответы и журнал проверки."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _arguments(argv)
    started = time.monotonic()
    input_directory = args.data_dir or data_dir()
    rows = load_questions(input_directory, args.questions)
    if args.limit:
        rows = rows[:args.limit]
    settings = LLMSettings.from_environment()
    service_class = _service_type(args.mode or os.getenv("QUERY_PROCESSING_MODE", "single"))
    retries = max_retries()
    corpus = Corpus.from_directory(input_directory)
    directory = artifacts_dir()
    team = create_search(corpus, directory)
    total_budget = min(1170.0, float(os.getenv("TIME_BUDGET_SEC", "1140")))

    def checkpoint(records: list[dict]) -> None:
        """Атомарно сохраняет CSV и трассу после каждого завершённого вопроса."""
        write_submission(records, args.out or Path.cwd() / "submission.csv")
        write_trace(records, args.trace or directory / "answer_trace.jsonl")

    try:
        records = asyncio.run(_process_rows(
            rows, settings, service_class, corpus, team, retries,
            checkpoint=checkpoint, time_budget=max(0.01, total_budget - time.monotonic() + started),
        ))
    finally:
        team.close()
    checkpoint(records)
    summary = dict(Counter(record["status"] for record in records))
    LOGGER.info("Итоги %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
