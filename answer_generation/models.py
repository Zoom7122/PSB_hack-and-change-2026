"""Строгие форматы результата генерации и смысловой проверки."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator


class Claim(BaseModel):
    """Одно утверждение и подтверждающие его идентификаторы фрагментов."""

    model_config = ConfigDict(extra="forbid")
    text: StrictStr = Field(min_length=1)
    chunk_ids: list[StrictStr] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def check_text(cls, value: str) -> str:
        """Отклоняет пустые утверждения."""
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value.strip()

    @field_validator("chunk_ids")
    @classmethod
    def check_ids(cls, value: list[str]) -> list[str]:
        """Требует непустые и неповторяющиеся идентификаторы источников."""
        if any(not item.strip() or item != item.strip() for item in value):
            raise ValueError("chunk IDs must not be blank or padded")
        if len(value) != len(set(value)):
            raise ValueError("duplicate chunk IDs")
        return value


class Draft(BaseModel):
    """Структура ответа модели до проверки и форматирования."""

    model_config = ConfigDict(extra="forbid")
    status: Literal["answered", "insufficient"]
    claims: list[Claim]
    reason: StrictStr | None = Field(default=None, max_length=800)

    @model_validator(mode="after")
    def check_status(self) -> "Draft":
        """Связывает наличие утверждений с выбранным статусом."""
        if (self.status == "answered") != bool(self.claims):
            raise ValueError("answered requires claims; insufficient requires none")
        return self


class ClaimCheck(BaseModel):
    """Оценка подтверждения одного утверждения отдельным вызовом модели."""

    model_config = ConfigDict(extra="forbid")
    claim_index: StrictInt = Field(ge=0)
    supported: StrictBool
    reason: StrictStr


class SemanticChecks(BaseModel):
    """Набор оценок для всех утверждений ответа."""

    model_config = ConfigDict(extra="forbid")
    checks: list[ClaimCheck]
