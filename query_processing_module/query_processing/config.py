"""Загрузка и проверка параметров OpenAI-совместимого API."""

import math
import os
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Некорректные или отсутствующие параметры подключения."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str = ""
    temperature: float | None = 0.0
    max_tokens: int | None = 1500
    timeout: float = 60.0

    def __post_init__(self) -> None:
        """Проверяет адрес, обязательные поля и диапазоны настроек."""
        try:
            parts = urlsplit(self.base_url)
            valid_url = parts.scheme in {"http", "https"} and bool(parts.hostname) and not parts.username and not parts.password and parts.port != 0
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ConfigurationError("LLM_BASE_URL must be an HTTP(S) URL without credentials")
        if not self.api_key.strip() or not self.model.strip():
            raise ConfigurationError("LLM_API_KEY and LLM_MODEL are required")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ConfigurationError("LLM_TIMEOUT must be positive")
        if self.temperature is not None and (not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2):
            raise ConfigurationError("LLM_TEMPERATURE must be between 0 and 2")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ConfigurationError("LLM_MAX_TOKENS must be positive")

    @classmethod
    def from_environment(cls) -> "LLMSettings":
        """Читает .env без переопределения переменных процесса."""
        load_dotenv(override=False)
        try:
            temperature = os.getenv("LLM_TEMPERATURE", "0").strip()
            max_tokens = os.getenv("LLM_MAX_TOKENS", "1500").strip()
            return cls(
                base_url=os.getenv("LLM_BASE_URL", "").strip(),
                api_key=os.getenv("LLM_API_KEY", "").strip(),
                model=os.getenv("LLM_MODEL", "").strip(),
                temperature=float(temperature) if temperature else None,
                max_tokens=int(max_tokens) if max_tokens else None,
                timeout=float(os.getenv("LLM_TIMEOUT", "60")),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError("Invalid numeric LLM configuration") from None
