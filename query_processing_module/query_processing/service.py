"""Основной сервис обработки запросов без зависимости от HTTP и Docker."""

import json

from .contracts import QueryRequest
from .llm_client import LLMClient
from .prompts import RETRY_PROMPT, SYSTEM_PROMPT
from .validator import InvalidLLMResponse, validate_response


class QueryProcessingService:
    def __init__(self, llm_client: LLMClient) -> None:
        """Принимает подменяемый клиент модели."""
        self._llm_client = llm_client

    async def process_query(self, request: dict | QueryRequest) -> dict:
        """Проверяет запрос, вызывает модель и возвращает результат для поиска."""
        query = QueryRequest.model_validate(request)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(query.model_dump(), ensure_ascii=False)},
        ]
        for attempt in range(2):
            content = await self._llm_client.complete(messages)
            try:
                return validate_response(content)
            except InvalidLLMResponse:
                if attempt == 1:
                    raise
                messages = [*messages, {"role": "user", "content": RETRY_PROMPT}]
        raise AssertionError("unreachable")
