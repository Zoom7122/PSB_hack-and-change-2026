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
        """Читает параметры платформы и локальный .env с приоритетом процесса."""
        process_environment = dict(os.environ)
        load_dotenv(override=False)
        dotenv_environment = dict(os.environ)
        try:
            base_url = _setting_value(
                ("LLM_BASE_URL", "OPENAI_BASE_URL"),
                process_environment,
                dotenv_environment,
            )
            api_key = _setting_value(
                ("LLM_API_KEY", "OPENAI_API_KEY"),
                process_environment,
                dotenv_environment,
            )
            model = _setting_value(
                ("LLM_MODEL", "OPENAI_MODEL", "GEN_MODEL"),
                process_environment,
                dotenv_environment,
            )
            temperature = _setting_value(
                ("TEMPERATURE", "LLM_TEMPERATURE"),
                process_environment,
                dotenv_environment,
                default="0",
            )
            max_tokens = _setting_value(
                ("MAX_TOKENS", "LLM_MAX_TOKENS"),
                process_environment,
                dotenv_environment,
                default="1500",
            )
            timeout = _setting_value(
                ("LLM_TIMEOUT",),
                process_environment,
                dotenv_environment,
                default="60",
            )
            return cls(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=float(temperature) if temperature else None,
                max_tokens=int(max_tokens) if max_tokens else None,
                timeout=float(timeout),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ConfigurationError):
                raise
            raise ConfigurationError("Invalid numeric LLM configuration") from None


def _setting_value(
    names: tuple[str, ...],
    process_environment: dict[str, str],
    dotenv_environment: dict[str, str],
    default: str = "",
) -> str:
    """Выбирает первый заданный alias, сохраняя приоритет переменных процесса."""
    for source in (process_environment, dotenv_environment):
        found_empty = False
        for name in names:
            if name not in source:
                continue
            value = source[name].strip()
            if value:
                return value
            found_empty = True
        if found_empty:
            return ""
    return default
