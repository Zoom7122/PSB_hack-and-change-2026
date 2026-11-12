"""HTTP-обёртка модуля 5 (Answer Generation + Verification)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from answer_generation.config import GenerationSettings, LLMSettings
from answer_generation.contracts import GenerationRequest, GenerationResult
from answer_generation.llm_client import OpenAIClient
from answer_generation.service import AnswerGenerationService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    client = OpenAIClient(LLMSettings.from_environment())
    app.state.answer_service = AnswerGenerationService(client, GenerationSettings.from_environment())
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Answer Generation + Verification", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Приложение запущено. Доступность LLM не проверяется."""
    return {"status": "ok", "module": "answer_generation"}


@app.post("/answer/generate", response_model=GenerationResult,
          responses={200: {"description": "GenerationResult; с ?debug=true — плюс поле debug"}})
async def generate(payload: GenerationRequest, request: Request,
                   debug: bool = Query(False, description="добавить трассировку этапов")):
    """GenerationRequest → GenerationResult (CONTRACTS-ALL.md, §5).

    Сбой модели не приводит к 5xx: возвращается status=insufficient с пояснением
    в verification.notes — так модуль 1 всегда получает валидный ответ.
    """
    outcome = await request.app.state.answer_service.run(payload)
    return JSONResponse(outcome.to_dict(include_debug=debug))
