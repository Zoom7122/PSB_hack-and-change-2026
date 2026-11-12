"""HTTP-обёртка над независимым сервисом подготовки запросов."""

from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException, Request

from query_processing.config import ConfigurationError, LLMSettings
from query_processing.contracts import QueryRequest, QueryResult
from query_processing.llm_client import LLMError, LLMTimeoutError, OpenAIClient
from query_processing.multi_stage import MultiStageQueryProcessingService
from query_processing.service import QueryProcessingService
from query_processing.validator import InvalidLLMResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Создаёт общий клиент при запуске и закрывает его при завершении."""
    settings = LLMSettings.from_environment()
    mode = os.getenv("QUERY_PROCESSING_MODE", "single").strip().lower()
    if mode not in {"single", "multi"}:
        raise ConfigurationError("QUERY_PROCESSING_MODE must be single or multi")
    client = OpenAIClient(settings)
    service_type = MultiStageQueryProcessingService if mode == "multi" else QueryProcessingService
    app.state.query_service = service_type(client)
    try:
        yield
    finally:
        await client.close()


app = FastAPI(title="Query Processing", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    """Показывает, что приложение запущено и обслуживает запросы."""
    return {"status": "ok"}


@app.post("/query/process", response_model=QueryResult)
async def process_query(query: QueryRequest, request: Request) -> dict:
    """Передаёт валидированный запрос сервису и отображает ошибки транспорта."""
    try:
        return await request.app.state.query_service.process_query(query)
    except LLMTimeoutError:
        raise HTTPException(status_code=504, detail="LLM request timed out") from None
    except (LLMError, InvalidLLMResponse):
        raise HTTPException(status_code=502, detail="LLM service failed") from None
