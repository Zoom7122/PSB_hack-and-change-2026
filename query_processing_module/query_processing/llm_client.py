"""Асинхронный транспорт для OpenAI-совместимых провайдеров."""

from typing import Protocol

import openai

from .config import LLMSettings


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
        """Отправляет сообщения модели и преобразует ошибки в безопасные исключения."""
        options = {}
        if self.settings.temperature is not None:
            options["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            options["max_tokens"] = self.settings.max_tokens
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.model, messages=messages, **options
            )
        except openai.APITimeoutError:
            raise LLMTimeoutError("LLM request timed out") from None
        except openai.APIError:
            raise LLMError("LLM provider request failed") from None
        if not response.choices or not response.choices[0].message.content:
            return ""
        return response.choices[0].message.content

    async def close(self) -> None:
        """Освобождает HTTP-ресурсы клиента."""
        await self._client.close()
