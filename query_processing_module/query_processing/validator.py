"""Проверка синтаксиса JSON и структуры ответа модели."""

from pydantic import ValidationError

from .contracts import QueryResult
from .json_utils import json_object


class InvalidLLMResponse(ValueError):
    """Ответ модели не соответствует согласованному контракту."""


def validate_response(content: str) -> dict:
    """Разбирает JSON и возвращает проверенный сериализуемый результат."""
    try:
        result = QueryResult.model_validate(json_object(content))
    except (ValidationError, ValueError):
        raise InvalidLLMResponse("LLM response does not match the expected JSON contract") from None
    return result.model_dump(mode="json")
