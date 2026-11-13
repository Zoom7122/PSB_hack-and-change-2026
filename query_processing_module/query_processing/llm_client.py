"""Асинхронный транспорт для OpenAI-совместимых провайдеров."""

from typing import Protocol
import asyncio
import logging
import os

import openai

from .config import LLMSettings

LOGGER = logging.getLogger(__name__)


class LLMError(Exception):
    """Ошибка доступа к поставщику модели без раскрытия деталей запроса."""


class LLMTimeoutError(LLMError):
    """Истекло время ожидания ответа поставщика модели."""


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Возвращает текст ответа на список сообщений."""
        ...


class OpenAIClient:
    def __init__(self, settings: LLMSettings) -> None:
        """Создаёт клиент с заданным таймаутом и без скрытых повторов SDK."""
        self.settings = settings
        self._client = openai.AsyncOpenAI(
            api_key=settings.api_key, base_url=settings.base_url,
            timeout=settings.timeout, max_retries=0,
        )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Повторяет временные ошибки сервиса, не раскрывая ключ и ответ API."""
        try:
            retries = min(3, max(0, int(os.getenv("LLM_MAX_RETRIES", "2"))))
        except ValueError:
            retries = 2
        for attempt in range(retries + 1):
            try:
                return await self._complete_once(messages)
            except openai.APIError as exc:
                status = getattr(exc, "status_code", None)
                temporary = isinstance(exc, openai.APIConnectionError) or status == 429 or (
                    isinstance(status, int) and status >= 500
                )
                if temporary and attempt < retries:
                    LOGGER.warning("Повтор LLM: type=%s status=%s attempt=%s",
                                   type(exc).__name__, status, attempt + 1)
                    await asyncio.sleep(min(4.0, 0.5 * 2 ** attempt))
                    continue
                if isinstance(exc, openai.APITimeoutError):
                    raise LLMTimeoutError("LLM request timed out") from None
                raise LLMError(f"LLM provider request failed (status={status})") from None
        raise AssertionError("unreachable")

    async def _complete_once(self, messages: list[dict[str, str]]) -> str:
        """Выполняет один вызов с параметрами генерации, переданными платформой."""
        options = {}
        if self.settings.temperature is not None:
            options["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            options["max_tokens"] = self.settings.max_tokens
        response = await self._client.chat.completions.create(
            model=self.settings.model, messages=messages, **options
        )
        if not response.choices or not response.choices[0].message.content:
            return ""
        return response.choices[0].message.content

    async def close(self) -> None:
        """Освобождает HTTP-ресурсы клиента."""
        await self._client.close()
