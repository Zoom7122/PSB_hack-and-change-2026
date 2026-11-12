"""Модуль 5 юридического RAG-ассистента: генерация ответа и проверка источников.

Быстрый старт (асинхронно, как модуль 2):

    from answer_generation import AnswerGenerationService, OpenAIClient, LLMSettings
    service = AnswerGenerationService(OpenAIClient(LLMSettings.from_environment()))
    result = await service.generate(generation_request_dict)

Синхронно (solution.py, потоки):

    from answer_generation import generate_answer_sync, SyncLLMAdapter
    result = generate_answer_sync(generation_request_dict, SyncLLMAdapter(main_llm_client))
"""

from __future__ import annotations

import asyncio
from typing import Any

from .config import ConfigurationError, GenerationSettings, LLMSettings
from .contracts import GenerationRequest, GenerationResult, RerankedChunk
from .llm_client import LLMClient, LLMError, LLMTimeoutError, OpenAIClient, SyncLLMAdapter
from .service import AnswerGenerationService, GenerationOutcome

__all__ = [
    "AnswerGenerationService", "GenerationOutcome", "GenerationRequest", "GenerationResult",
    "RerankedChunk", "GenerationSettings", "LLMSettings", "ConfigurationError", "LLMClient",
    "LLMError", "LLMTimeoutError", "OpenAIClient", "SyncLLMAdapter", "generate_answer_sync",
]


def generate_answer_sync(request: dict | GenerationRequest, llm_client: LLMClient,
                         settings: GenerationSettings | None = None, *,
                         include_debug: bool = False) -> dict[str, Any]:
    """Синхронная обёртка: безопасна в рабочих потоках ThreadPoolExecutor.

    В потоке без event loop использует asyncio.run. Внутри работающего loop
    вызывать нельзя — там используйте `await service.generate(...)`.
    """
    service = AnswerGenerationService(llm_client, settings)

    async def _run() -> dict[str, Any]:
        outcome = await service.run(request)
        return outcome.to_dict(include_debug=include_debug)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())
    raise RuntimeError("generate_answer_sync called inside a running event loop; use await service.generate()")
