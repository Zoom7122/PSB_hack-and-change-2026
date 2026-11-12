"""Контракты входного запроса и результата Query Processing."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["user", "assistant"]
    content: StrictStr = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        """Запрещает сообщения, состоящие только из пробелов."""
        if not value.strip():
            raise ValueError("content must not be blank")
        return value


class QueryRequest(StrictModel):
    question: StrictStr = Field(min_length=1)
    history: list[Message] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        """Проверяет, что вопрос содержит текст."""
        if not value.strip():
            raise ValueError("question must not be blank")
        return value


class Filters(StrictModel):
    document_number: StrictStr | None
    document_type: StrictStr | None
    event_date: date | None

    @field_validator("document_number", "document_type")
    @classmethod
    def filter_not_blank(cls, value: str | None) -> str | None:
        """Запрещает пустые строки вместо отсутствующих фильтров."""
        if value is not None and not value.strip():
            raise ValueError("filter must be non-empty or null")
        return value

    @field_validator("event_date", mode="before")
    @classmethod
    def validate_date_format(cls, value: object) -> object:
        """Разрешает дату только в формате YYYY-MM-DD или null."""
        if value is not None and (not isinstance(value, str) or len(value) != 10):
            raise ValueError("event_date must be YYYY-MM-DD or null")
        return value


class SearchRequest(StrictModel):
    standalone_query: StrictStr = Field(min_length=1)
    search_queries: list[StrictStr] = Field(min_length=1, max_length=3)
    filters: Filters

    @field_validator("standalone_query")
    @classmethod
    def standalone_not_blank(cls, value: str) -> str:
        """Проверяет содержательность самостоятельного вопроса."""
        if not value.strip():
            raise ValueError("standalone_query must not be blank")
        return value

    @field_validator("search_queries")
    @classmethod
    def searches_not_blank(cls, values: list[str]) -> list[str]:
        """Проверяет, что каждая поисковая формулировка содержит текст."""
        if any(not value.strip() for value in values):
            raise ValueError("search queries must not be blank")
        return values


class QueryResult(StrictModel):
    status: Literal["ready", "clarification_required", "out_of_scope"]
    search_request: SearchRequest | None
    message: StrictStr | None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "QueryResult":
        """Согласует наличие поискового запроса и сообщения со статусом."""
        if self.status == "ready":
            if self.search_request is None or self.message is not None:
                raise ValueError("ready requires search_request and null message")
        elif self.search_request is not None or not self.message or not self.message.strip():
            raise ValueError("non-ready requires null search_request and a message")
        return self
