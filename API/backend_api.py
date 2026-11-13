"""Нормализация и валидация входа для Query Processing."""

import re

from query_processing_module.query_processing.contracts import QueryRequest

ROLE = re.compile(r"(?im)^\s*(Пользователь|Ассистент|User|Assistant)\s*:\s*")


def prepare_query(question: str) -> QueryRequest:
    """Отделяет последнюю реплику CSV-диалога от его собственной истории."""
    text = question.replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(ROLE.finditer(text))
    if len(matches) < 2 or text[:matches[0].start()].strip():
        return QueryRequest.model_validate({"question": text, "history": []})
    messages = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end():end].strip()
        role = "user" if match.group(1).casefold() in {"пользователь", "user"} else "assistant"
        messages.append({"role": role, "content": content})
    if messages[-1]["role"] != "user":
        return QueryRequest.model_validate({"question": text, "history": []})
    return QueryRequest.model_validate(
        {"question": messages[-1]["content"], "history": messages[:-1]}
    )
