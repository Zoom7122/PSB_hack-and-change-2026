"""Платформенный CSV-пайплайн подготовки, поиска и проверки ответов."""

import asyncio
import csv
import json
import logging
import os
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
        LOGGER.info("LLM response %s", json.dumps({
            "id": self._row_id, "call": self._call_number,
            "response": _redact_secret(response, self._api_key),
        }, ensure_ascii=False))
        return response


def _redact_secret(value: str, secret: str) -> str:
    """Заменяет точные вхождения API-ключа перед записью в лог."""
    return value.replace(secret, "[REDACTED_SECRET]") if secret else value


def load_questions(directory: Path) -> list[QuestionRow]:
    """Читает CSV, проверяет ID и формирует запросы по контракту."""
    with (directory / "questions.csv").open(encoding="utf-8-sig", newline="") as source:
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


async def _run_row(row: QuestionRow, client: LLMClient, service_class: type[QueryService],
                   corpus: Corpus, team, retries: int) -> dict:
    """Обрабатывает один вопрос от Query Processing до проверенного ответа."""
    base = {
        "id": row.row_id, "question": row.original_question,
        "corpus_version": corpus.version, "search_request": None,
    }
    try:
        query_result = await service_class(client).process_query(row.request)
        status = query_result["status"]
        if status != "ready":
            result = _early_result(status, query_result["message"])
        else:
            request = query_result["search_request"]
            base["search_request"] = request
            sources = await asyncio.to_thread(retrieve, corpus, team, row.original_question, request)
            result = await generate_verified(
                client, row.original_question, sources, corpus,
                event_date=request.get("filters", {}).get("event_date"), retries=retries,
            )
        LOGGER.info("Обработка завершена id=%s status=%s", row.row_id, result["status"])
    except Exception as exc:
        LOGGER.error("Ошибка запроса id=%s type=%s", row.row_id, type(exc).__name__)
        result = _early_result("insufficient", INSUFFICIENT, type(exc).__name__)
    return {**base, **result}


async def _process_rows(rows: list[QuestionRow], settings: LLMSettings,
                        service_class: type[QueryService], corpus: Corpus,
                        team, retries: int) -> list[dict]:
    """Обрабатывает вопросы параллельно и закрывает общий LLM-клиент."""
    client = OpenAIClient(settings)
    semaphore = asyncio.Semaphore(_worker_count())

    async def process(row: QuestionRow) -> dict:
        """Ограничивает число одновременно обрабатываемых вопросов."""
        async with semaphore:
            wrapped = LoggingLLMClient(client, row.row_id, settings.api_key)
            return await _run_row(row, wrapped, service_class, corpus, team, retries)

    try:
        return await asyncio.gather(*(process(row) for row in rows))
    finally:
        await client.close()


def main() -> None:
    """Запускает поиск и генерацию, затем сохраняет ответы и журнал проверки."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = load_questions(data_dir())
    settings = LLMSettings.from_environment()
    service_class = _service_type(os.getenv("QUERY_PROCESSING_MODE", "single"))
    retries = max_retries()
    corpus = Corpus.from_directory(data_dir())
    directory = artifacts_dir()
    team = create_search(corpus, directory)
    try:
        records = asyncio.run(_process_rows(rows, settings, service_class, corpus, team, retries))
    finally:
        team.close()
    write_trace(records, directory / "answer_trace.jsonl")
    write_submission(records, Path.cwd() / "submission.csv")


if __name__ == "__main__":
    main()
