"""Резервный модульный путь, используемый при сбое платформенного ответа."""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from API.backend_api import prepare_query
from API.for_solutions import QuestionRow, _run_row, _service_type
from answer_generation.service import max_retries
from query_processing_module.query_processing.config import LLMSettings
from query_processing_module.query_processing.llm_client import OpenAIClient
from rag_pipeline.corpus import Corpus
import numpy as np

from data_processing import hybrid, search, vector_index
from rag_pipeline.indexing import TeamSearch

LOG = logging.getLogger(__name__)


async def _execute(question: str, directory: Path, timeout: float) -> dict:
    """Запускает подготовку, поиск и проверенную генерацию с общим лимитом."""
    settings = LLMSettings.from_environment()
    corpus = await asyncio.to_thread(Corpus.from_directory, directory)
    vectors = vector_index.VectorIndex(corpus.chunks, np.zeros((len(corpus.chunks), 1)))
    team = TeamSearch(hybrid.HybridSearcher(search.build_index(corpus.chunks), vectors), None)
    client = OpenAIClient(settings)
    row = QuestionRow("fallback", prepare_query(question), question)
    try:
        return await asyncio.wait_for(
            _run_row(row, client, _service_type("single"), corpus, team, max_retries()),
            timeout=max(0.01, timeout),
        )
    finally:
        await client.close()
        team.close()


def recover(question: str, directory: Path, deadline: float) -> dict | None:
    """Возвращает только подтверждённый модульный ответ при достаточном запасе времени."""
    remaining = deadline - time.monotonic()
    if remaining < 30:
        return None
    try:
        result = asyncio.run(_execute(question, directory, min(remaining - 5, 90)))
        return result if result.get("status") == "answered" else None
    except Exception as exc:
        LOG.warning("Резервный конвейер недоступен: %s", type(exc).__name__)
        return None
