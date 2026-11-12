"""Контракт модуля 5: GenerationRequest → GenerationResult (CONTRACTS-ALL.md, §5).

Принцип стыковки
================
* Вход принимаем терпимо: модуль 4 может добавлять свои поля (`cite_label`,
  `scores`, `rrf_rank`, …) и отдавать `null` в реквизитах, которых нет в базе
  (`revision_id`, `page`, `source_path`). Лишние поля чанка сохраняются
  (extra="allow") — часть из них мы используем (`cite_label`, `scores`).
* Выход строгий: ровно поля контракта, инварианты проверяются валидатором.
  Мы скорее упадём в тестах, чем отдадим модулю 1 «answered» без проверки.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator


def _lenient_date(value: Any) -> Any:
    """Дата YYYY-MM-DD или None. Мусор от соседнего модуля не роняет запрос."""
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        text = value.strip()[:10]
        try:
            return date.fromisoformat(text)
        except ValueError:
            return None
    return None


class RerankedChunk(BaseModel):
    """Чанк от Reranker. Обязательны только идентификатор и текст."""

    model_config = ConfigDict(extra="allow")

    chunk_id: StrictStr = Field(min_length=1)
    text: str
    document_id: str | None = None
    document_title: str | None = None
    document_type: str | None = None
    document_number: str | None = None
    revision_id: str | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    section: str | None = None
    page: int | None = None
    source_path: str | None = None
    rank: int | None = None
    rerank_score: float | None = None

    @field_validator("effective_from", "effective_to", mode="before")
    @classmethod
    def _lenient_dates(cls, value: Any) -> Any:
        return _lenient_date(value)

    @field_validator("page", mode="before")
    @classmethod
    def _lenient_page(cls, value: Any) -> Any:
        if value is None or isinstance(value, int):
            return value
        try:
            return int(str(value).strip())
        except ValueError:
            return None

    def extra_field(self, name: str, default: Any = None) -> Any:
        return (self.model_extra or {}).get(name, default)


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    request_id: StrictStr = Field(min_length=1)
    corpus_version: str = ""
    question: StrictStr = Field(min_length=1)
    event_date: date | None = None
    chunks: list[RerankedChunk] = Field(default_factory=list)

    @field_validator("event_date", mode="before")
    @classmethod
    def _lenient_event_date(cls, value: Any) -> Any:
        return _lenient_date(value)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value

    @field_validator("corpus_version", mode="before")
    @classmethod
    def _corpus_none(cls, value: Any) -> Any:
        return "" if value is None else value


# --------------------------------------------------------------------- выход
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Claim(StrictModel):
    claim_id: StrictStr
    text: StrictStr
    chunk_ids: list[StrictStr]
    support: Literal["supported", "unsupported"]


class CitationLink(StrictModel):
    chunk_id: StrictStr
    claim_ids: list[StrictStr]


class Verification(StrictModel):
    status: Literal["passed", "failed", "not_applicable"]
    invalid_chunk_ids: list[StrictStr] = Field(default_factory=list)
    unsupported_claim_ids: list[StrictStr] = Field(default_factory=list)
    notes: list[StrictStr] = Field(default_factory=list)


class GenerationResult(StrictModel):
    request_id: StrictStr
    status: Literal["answered", "insufficient"]
    answer: StrictStr = Field(min_length=1)
    claims: list[Claim] = Field(default_factory=list)
    citations: list[CitationLink] = Field(default_factory=list)
    verification: Verification

    @model_validator(mode="after")
    def _invariants(self) -> "GenerationResult":
        claim_ids = {claim.claim_id for claim in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("claim_id must be unique")
        if self.status == "answered":
            if self.verification.status != "passed":
                raise ValueError("answered requires verification.status=passed")
            if not self.claims or any(c.support != "supported" for c in self.claims):
                raise ValueError("answered requires only supported claims")
        supported = {c.claim_id for c in self.claims if c.support == "supported"}
        cited_chunks: set[str] = set()
        for link in self.citations:
            if link.chunk_id in cited_chunks:
                raise ValueError("citations must not repeat chunk_id")
            cited_chunks.add(link.chunk_id)
            if not link.claim_ids or not set(link.claim_ids) <= supported:
                raise ValueError("citations may reference only supported claims")
        for claim in self.claims:
            if claim.support == "supported" and not claim.chunk_ids:
                raise ValueError("supported claim must reference chunks")
        return self
