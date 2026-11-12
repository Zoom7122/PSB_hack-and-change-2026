"""Настройки модуля 5: подключение к LLM и параметры генерации/проверки.

Имена переменных подключения совпадают с модулем 2 (Query Processing):
LLM_BASE_URL, LLM_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
LLM_TIMEOUT. Поэтому оба модуля читают один и тот же .env и могут делить
один клиент.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Некорректные или отсутствующие параметры."""


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str = ""
    temperature: float | None = 0.0
    max_tokens: int | None = 1500
    timeout: float = 60.0
    json_mode: bool = False

    def __post_init__(self) -> None:
        try:
            parts = urlsplit(self.base_url)
            valid_url = (
                parts.scheme in {"http", "https"}
                and bool(parts.hostname)
                and not parts.username
                and not parts.password
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ConfigurationError("LLM_BASE_URL must be an HTTP(S) URL without credentials")
        if not self.api_key.strip() or not self.model.strip():
            raise ConfigurationError("LLM_API_KEY and LLM_MODEL are required")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ConfigurationError("LLM_TIMEOUT must be positive")
        if self.temperature is not None and not 0 <= self.temperature <= 2:
            raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigurationError("LLM_MAX_TOKENS must be positive")

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        """Читает .env без переопределения переменных процесса."""
        load_dotenv(override=False)
        try:
            temperature = _env("LLM_TEMPERATURE", "0")
            max_tokens = _env("LLM_MAX_TOKENS", "1500")
            return cls(
                base_url=_env("LLM_BASE_URL", ""),
                api_key=_env("LLM_API_KEY", ""),
                model=_env("LLM_MODEL", ""),
                temperature=float(temperature) if temperature else None,
                max_tokens=int(max_tokens) if max_tokens else None,
                timeout=float(_env("LLM_TIMEOUT", "60")),
                json_mode=_env_bool("LLM_JSON_MODE", False),
            )
        except ConfigurationError:
            raise
        except (TypeError, ValueError):
            raise ConfigurationError("Invalid numeric LLM configuration") from None


@dataclass(frozen=True)
class GenerationSettings:
    """Поведение генерации и проверки. Значения по умолчанию — для платформы."""

    # Регламент платформы: не более 150 слов в ответе.
    max_answer_words: int = 150
    # Сколько чанков отдаём модели (вход уже отсортирован Reranker'ом).
    max_context_chunks: int = 8
    # Длинные статьи законов режем, чтобы не раздувать промпт.
    max_chunk_chars: int = 3000
    # "llm" — правила + LLM-проверка каждого утверждения; "rules" — только правила.
    verification_mode: str = "llm"
    # Сколько раз переписывать черновик по замечаниям проверки.
    max_repairs: int = 1
    # Сколько раз повторить вызов LLM при сетевой ошибке/таймауте.
    llm_retries: int = 1
    # Отказ без вызова LLM, если все чанки несут scores и они ниже порогов.
    evidence_gate: bool = True
    gate_bm25_min: float = 4.0
    gate_vector_min: float = 0.30
    # Если event_date не передан, пробуем извлечь дату события из вопроса.
    infer_event_date: bool = True

    def __post_init__(self) -> None:
        if self.verification_mode not in {"llm", "rules"}:
            raise ConfigurationError("ANSWER_VERIFICATION_MODE must be llm or rules")
        if self.max_answer_words < 20:
            raise ConfigurationError("ANSWER_MAX_WORDS must be at least 20")
        if self.max_context_chunks < 1 or self.max_chunk_chars < 200:
            raise ConfigurationError("context limits are too small")
        if self.max_repairs < 0 or self.llm_retries < 0:
            raise ConfigurationError("retry counters must be non-negative")

    @classmethod
    def from_environment(cls) -> "GenerationSettings":
        load_dotenv(override=False)
        try:
            return cls(
                max_answer_words=int(_env("ANSWER_MAX_WORDS", "150")),
                max_context_chunks=int(_env("ANSWER_MAX_CONTEXT_CHUNKS", "8")),
                max_chunk_chars=int(_env("ANSWER_MAX_CHUNK_CHARS", "3000")),
                verification_mode=_env("ANSWER_VERIFICATION_MODE", "llm").lower(),
                max_repairs=int(_env("ANSWER_MAX_REPAIRS", "1")),
                llm_retries=int(_env("ANSWER_LLM_RETRIES", "1")),
                evidence_gate=_env_bool("ANSWER_EVIDENCE_GATE", True),
                gate_bm25_min=float(_env("ANSWER_GATE_BM25_MIN", "4.0")),
                gate_vector_min=float(_env("ANSWER_GATE_VECTOR_MIN", "0.30")),
                infer_event_date=_env_bool("ANSWER_INFER_EVENT_DATE", True),
            )
        except ConfigurationError:
            raise
        except (TypeError, ValueError):
            raise ConfigurationError("Invalid numeric ANSWER_* configuration") from None
