"""Транспорт к LLM.

Протокол тот же, что в модуле 2: `async complete(messages) -> str`.
Любой объект с таким методом подходит — в том числе клиент модуля 2.

Для синхронного кода (solution.py из ветки main работает потоками и вызывает
`client.chat(messages, temperature=...)`) есть `SyncLLMAdapter`.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Callable, Protocol

from .config import LLMSettings


class LLMError(Exception):
    """Ошибка доступа к модели без раскрытия деталей запроса."""


class LLMTimeoutError(LLMError):
    """Истекло время ожидания ответа модели."""


class LLMClient(Protocol):
    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Возвращает текст ответа модели."""
        ...


class OpenAIClient:
    """OpenAI-совместимый клиент (DeepSeek, прокси платформы, OpenRouter…)."""

    def __init__(self, settings: LLMSettings) -> None:
        import openai  # импорт здесь: ядро модуля не зависит от SDK

        self._openai = openai
        self.settings = settings
        self._json_mode = settings.json_mode
        self._client = openai.AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
            max_retries=0,
        )

    async def complete(self, messages: list[dict[str, str]]) -> str:
        options: dict[str, Any] = {}
        if self.settings.temperature is not None:
            options["temperature"] = self.settings.temperature
        if self.settings.max_tokens is not None:
            options["max_tokens"] = self.settings.max_tokens
        if self._json_mode:
            options["response_format"] = {"type": "json_object"}
        openai = self._openai
        try:
            response = await self._client.chat.completions.create(
                model=self.settings.model, messages=messages, **options
            )
        except openai.BadRequestError:
            if not self._json_mode:
                raise LLMError("LLM provider rejected the request") from None
            # Провайдер не поддерживает response_format — работаем без него.
            self._json_mode = False
            return await self.complete(messages)
        except openai.APITimeoutError:
            raise LLMTimeoutError("LLM request timed out") from None
        except openai.APIError:
            raise LLMError("LLM provider request failed") from None
        if not response.choices or not response.choices[0].message.content:
            return ""
        return response.choices[0].message.content

    async def close(self) -> None:
        await self._client.close()


class SyncLLMAdapter:
    """Оборачивает синхронный клиент в протокол `complete`.

    Принимает:
      * объект с методом `chat(messages, temperature=...)` (llm_client.py из main);
      * объект с синхронным `complete(messages)`;
      * просто функцию `fn(messages) -> str`.
    Вызов уходит в поток, чтобы не блокировать event loop.
    """

    def __init__(self, target: Any, temperature: float | None = 0.0) -> None:
        self._call = self._resolve(target, temperature)

    @staticmethod
    def _resolve(target: Any, temperature: float | None) -> Callable[[list[dict[str, str]]], str]:
        if hasattr(target, "chat") and callable(target.chat):
            if temperature is None:
                return lambda messages: target.chat(messages)
            return lambda messages: target.chat(messages, temperature=temperature)
        if hasattr(target, "complete") and callable(target.complete):
            if inspect.iscoroutinefunction(target.complete):
                raise TypeError("async client: pass it to the service directly, без адаптера")
            return target.complete
        if callable(target):
            return target
        raise TypeError("target must be callable or expose chat()/complete()")

    async def complete(self, messages: list[dict[str, str]]) -> str:
        try:
            result = await asyncio.to_thread(self._call, messages)
        except TimeoutError as exc:
            raise LLMTimeoutError(str(exc) or "LLM request timed out") from None
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001 — чужой клиент, приводим к нашему типу
            raise LLMError(f"{type(exc).__name__}") from None
        return result or ""
