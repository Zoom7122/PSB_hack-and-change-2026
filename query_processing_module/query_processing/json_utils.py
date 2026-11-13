"""Разбор одного JSON-объекта из текстового ответа модели."""

import json
import re


def _unique_object(pairs: list[tuple]) -> dict:
    """Отклоняет неоднозначные JSON-объекты с повторяющимися ключами."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def json_object(text: str) -> dict:
    """Принимает JSON и Markdown-ограждение, сохраняя строгую проверку схемы."""
    value = text.lstrip("\ufeff").strip()
    value = re.sub(r"^<think>.*?</think>\s*", "", value, flags=re.DOTALL)
    if value.startswith(chr(96) * 3):
        value = re.sub(r"^\x60{3}(?:json)?\s*", "", value, count=1, flags=re.IGNORECASE)
        value = re.sub(r"\s*\x60{3}$", "", value, count=1).strip()
    result = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(result, dict):
        raise ValueError("LLM response must contain one JSON object")
    return result
