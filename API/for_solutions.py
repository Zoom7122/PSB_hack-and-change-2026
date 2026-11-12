"""CSV-обвязка решения с прямым вызовом сервиса Query Processing."""

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
from query_processing_module.query_processing.llm_client import (
    LLMClient,
    OpenAIClient,
)
from query_processing_module.query_processing.multi_stage import (
    MultiStageQueryProcessingService,
)
from query_processing_module.query_processing.service import QueryProcessingService

from API.backend_api import prepare_query

LOGGER = logging.getLogger("solution")
SERVICE_TYPES = {
    "single": QueryProcessingService,
    "multi": MultiStageQueryProcessingService,
}


class QueryService(Protocol):
    """Описывает общий вход сервисов single- и multi-stage режимов."""

    async def process_query(self, request: dict | QueryRequest) -> dict:
        """Обрабатывает один валидированный запрос."""
        ...


@dataclass(frozen=True)
class QuestionRow:
    """Хранит исходный ID и проверенный запрос одной CSV-строки."""

    row_id: str
    request: QueryRequest


class LoggingLLMClient:
    """Логирует сырые ответы модели и возвращает их без изменения."""

    def __init__(self, client: LLMClient, row_id: str, api_key: str) -> None:
        """Связывает общий LLM-клиент с ID текущего вопроса."""
        self._client = client
        self._row_id = row_id
        self._api_key = api_key
        self._call_number = 0

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Запрашивает модель и пишет её ответ в безопасный JSON-лог."""
        response = await self._client.complete(messages)
        self._call_number += 1
        safe_response = _redact_secret(response, self._api_key)
        LOGGER.info(
            "LLM response %s",
            json.dumps(
                {
                    "id": self._row_id,
                    "call": self._call_number,
                    "response": safe_response,
                },
                ensure_ascii=False,
            ),
        )
        return response


def _redact_secret(value: str, secret: str) -> str:
    """Заменяет точные вхождения API-ключа перед записью в лог."""
    if not secret:
        return value
    return value.replace(secret, "[REDACTED_SECRET]")


def load_questions(data_dir: Path) -> list[QuestionRow]:
    """Читает CSV, проверяет ID и формирует запросы по контракту."""
    questions_path = data_dir / "questions.csv"
    with questions_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required_columns = {"id", "question"}
        if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
            raise ValueError("В questions.csv нужны колонки id и question")

        rows = []
        seen_ids = set()
        for line_number, row in enumerate(reader, start=2):
            row_id = row.get("id")
            if row_id is None or not row_id.strip():
                raise ValueError(f"В questions.csv пустой id в строке {line_number}")
            if row_id in seen_ids:
                raise ValueError(f"Повторяющийся id в questions.csv: {row_id}")
            seen_ids.add(row_id)
            question = row.get("question")
            if question is None:
                raise ValueError(f"Отсутствует question в строке {line_number}")
            rows.append(QuestionRow(row_id, prepare_query(question)))

    if not rows:
        raise ValueError("В questions.csv нет вопросов")
    return rows


def write_submission(rows: list[QuestionRow], path: Path) -> None:
    """Создаёт submission.csv с ID и пустыми полями answer."""
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(
            destination,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writerow(["id", "answer"])
        writer.writerows((row.row_id, "") for row in rows)


def _service_type(mode: str) -> type[QueryService]:
    """Выбирает реализацию Query Processing по переменной режима."""
    normalized_mode = mode.strip().lower()
    try:
        return SERVICE_TYPES[normalized_mode]
    except KeyError:
        raise ValueError("QUERY_PROCESSING_MODE must be single or multi") from None


def _worker_count() -> int:
    """Читает MAX_WORKERS и ограничивает параллельность диапазоном 1–8."""
    try:
        return min(8, max(1, int(os.getenv("MAX_WORKERS", "2"))))
    except ValueError:
        return 2


async def _process_row(
    row: QuestionRow,
    client: OpenAIClient,
    service_class: type[QueryService],
    api_key: str,
    semaphore: asyncio.Semaphore,
) -> None:
    """Выполняет один запрос и безопасно регистрирует его результат или ошибку."""
    async with semaphore:
        llm_client = LoggingLLMClient(client, row.row_id, api_key)
        service = service_class(llm_client)
        try:
            result = await service.process_query(row.request)
            LOGGER.info(
                "Query Processing завершён id=%s status=%s",
                row.row_id,
                result.get("status", "unknown"),
            )
        except Exception as exc:
            LOGGER.error(
                "Ошибка запроса id=%s type=%s",
                row.row_id,
                type(exc).__name__,
            )


async def _process_rows(
    rows: list[QuestionRow],
    settings: LLMSettings,
    service_class: type[QueryService],
) -> None:
    """Обрабатывает вопросы параллельно и закрывает общий LLM-клиент."""
    client = OpenAIClient(settings)
    semaphore = asyncio.Semaphore(_worker_count())
    try:
        await asyncio.gather(
            *(
                _process_row(
                    row,
                    client,
                    service_class,
                    settings.api_key,
                    semaphore,
                )
                for row in rows
            )
        )
    finally:
        await client.close()


def main() -> None:
    """Загружает вопросы, создаёт пустой сабмит и запускает Query Processing."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    data_dir = Path(os.getenv("DATA_DIR", "data").strip() or "data")
    rows = load_questions(data_dir)
    write_submission(rows, Path.cwd() / "submission.csv")

    settings = LLMSettings.from_environment()
    mode = os.getenv("QUERY_PROCESSING_MODE", "single")
    service_class = _service_type(mode)
    asyncio.run(_process_rows(rows, settings, service_class))
