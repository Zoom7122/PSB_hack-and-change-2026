"""Нормализация и валидация входа для Query Processing."""

from query_processing_module.query_processing.contracts import QueryRequest


def prepare_query(question: str) -> QueryRequest:
    """Нормализует текст и собирает проверенный запрос Query Processing."""
    normalized_question = question.strip().lower()
    return QueryRequest.model_validate(
        {"question": normalized_question, "history": []}
    )
