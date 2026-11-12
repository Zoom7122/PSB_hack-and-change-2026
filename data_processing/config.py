"""Настройки проекта: пути, доступ к моделям, параметры поиска.

Главная идея этого файла
========================
Код должен одинаково работать в двух средах:

  1) ЛОКАЛЬНО (разработка) — ваш собственный API: DeepSeek для генерации,
     OpenAI для эмбеддингов, реранкера может не быть вовсе;
  2) НА ПЛАТФОРМЕ (проверка) — один прокси, который даёт DeepSeek V4.1 Flash
     для генерации, bge-m3 для эмбеддингов и Qwen3 Reranker 8B для реранжа.

Поэтому у нас ТРИ независимые группы настроек: генерация, эмбеддинги, реранк.
Каждая может смотреть на свой адрес и свой ключ. На платформе все три будут
указывать на один прокси и различаться только именами моделей.

Чего делать НЕЛЬЗЯ
==================
* хардкодить имя модели — на платформе оно приходит в LLM_MODEL / EMB_MODEL;
* хардкодить размерность эмбеддингов — у bge-m3 (1024) и у OpenAI small (1536)
  она разная, и индекс, собранный локально, не подойдёт к платформе;
* класть ключи в код или в файлы, попадающие в ZIP-сабмит.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _resolve_dir(env_name: str, default: str) -> Path:
    """Каталог из окружения; относительный путь считаем от корня проекта."""
    raw = os.getenv(env_name, "").strip()
    path = Path(raw or default)
    return path if path.is_absolute() else (BASE_DIR / path)


def _env(*names: str, default: str = "") -> str:
    """Первое непустое значение среди перечисленных имён переменных."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _load_dotenv(path: Path | None = None) -> None:
    """Прочитать .env, если он есть (локальная разработка).

    В .env лежат ключи. В git его не коммитим — см. .gitignore.
    На платформе .env нет: переменные подставит окружение контейнера.
    """
    file = path or (BASE_DIR / ".env")
    if not file.is_file():
        return
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# --- пути -------------------------------------------------------------------
DATA_DIR = _resolve_dir("DATA_DIR", "data")
ARTIFACTS_DIR = _resolve_dir("ARTIFACTS_DIR", "artifacts")

# --- генерация --------------------------------------------------------------
LLM_BASE_URL = _env("LLM_BASE_URL", "OPENAI_BASE_URL").rstrip("/")
LLM_API_KEY = _env("LLM_API_KEY", "OPENAI_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL")

# --- эмбеддинги -------------------------------------------------------------
# Отдельные адрес и ключ: локально это OpenAI, на платформе — прокси с bge-m3.
# Если не заданы, используем значения генерации (на платформе они совпадают).
EMB_BASE_URL = (_env("EMB_BASE_URL", "EMBEDDER_BASE_URL") or LLM_BASE_URL).rstrip("/")
EMB_API_KEY = _env("EMB_API_KEY", "EMBEDDER_API_KEY") or LLM_API_KEY
EMB_MODEL = _env("EMB_MODEL", default="baai/bge-m3")
EMB_BATCH_SIZE = int(_env("EMB_BATCH_SIZE", default="32"))

# Префикс-инструкция для ЗАПРОСА (не для документов).
# Некоторые модели обучены с инструкцией: качество поиска заметно выше, если
# запрос оформлен по правилам модели. Особенно это важно для Qwen3-Embedding —
# без инструкции на запросе он теряет несколько пунктов hit@1.
# Документы при этом кодируются БЕЗ префикса — это требование модели.
EMB_QUERY_PREFIX = _env("EMB_QUERY_PREFIX", default="")

# --- реранк -----------------------------------------------------------------
RERANK_BASE_URL = (_env("RERANK_BASE_URL", "RERANKER_BASE_URL") or LLM_BASE_URL).rstrip("/")
RERANK_API_KEY = _env("RERANK_API_KEY", "RERANKER_API_KEY") or LLM_API_KEY
RERANK_MODEL = _env("RERANKER_MODEL", "RERANK_MODEL", default="")

# Режим реранжирования. Три варианта, выбираются автоматически или вручную:
#   "endpoint" — POST /rerank на провайдере (Qwen3 Reranker на платформе);
#   "llm"      — список фрагментов отдаётся языковой модели, она возвращает порядок
#                (работает с любым chat-API, но дороже и медленнее);
#   "none"     — без реранжа, порядок после RRF.
# auto = endpoint, если задан RERANK_MODEL, иначе llm, если есть ключ, иначе none.
RERANK_MODE = _env("RERANK_MODE", default="auto")
RERANK_TOP_N = int(_env("RERANKER_TOP_N", default="6"))

# --- параметры поиска -------------------------------------------------------
BM25_TOP_K = int(_env("TOP_K", default="30"))       # кандидатов из лексики
VECTOR_TOP_K = int(_env("VECTOR_TOP_K", "TOP_K", default="30"))
RRF_K = 60                                          # сглаживание в RRF

# Слияние неравноправное: если один поиск заметно сильнее на нашей базе, его
# голос должен весить больше. Вес 0.3 означает «одно попадание BM25 стоит
# примерно как треть попадания векторов». Значение подбирается измерением —
# см. tune_fusion.py.
BM25_WEIGHT = float(_env("BM25_WEIGHT", default="0.1"))
# BM25 надёжен в начале списка и шумит в хвосте: 20-е место у него часто
# случайное совпадение по одному общему слову. Учитываем только первые N мест.
BM25_FUSION_DEPTH = int(_env("BM25_FUSION_DEPTH", default="5"))
RERANK_CANDIDATES = int(_env("RERANK_CANDIDATES", default="15"))

# --- генерация: параметры ---------------------------------------------------
# --- ответ (шаг 6) ----------------------------------------------------------
# Пороги отказа взяты из измерений шага 3: у вопроса без ответа лучший BM25 был
# 2.37, у отвечаемых — от 8.20. Ставим 4.0: вдвое выше «пустого» и вдвое ниже
# минимального «нормального».
REFUSAL_BM25_MIN = float(_env("REFUSAL_BM25_MIN", default="4.0"))
# Сходство векторов: отказ только если и оно низкое. 0.30 — заведомо ниже любых
# осмысленных совпадений на нашей базе, порог страхует от ложных отказов.
REFUSAL_VECTOR_MIN = float(_env("REFUSAL_VECTOR_MIN", default="0.30"))
# Регламент: не более 150 слов на ответ и 10 000 символов на строку CSV.
ANSWER_MAX_WORDS = int(_env("ANSWER_MAX_WORDS", default="150"))
# Как оформлять ссылки в итоговом ответе:
#   inline  — заменить [n] на «(п. 3.1)» прямо в тексте (по умолчанию,
#             так оформлены примеры ответов организаторов в demo_output.csv);
#   sources — убрать [n] и дописать «Источники: <документ>, <пункт>» отдельной строкой;
#   markers — оставить [n] как есть (нужно, если проверяющий ждёт номера);
#   short   — одна строка «факт — документ, пункт» (самый компактный вариант).
ANSWER_CITE_STYLE = _env("ANSWER_CITE_STYLE", default="inline")
# Жёсткий лимит длины ответа в символах. Часть грейдеров ограничивает поле ответа;
# 600 символов — это примерно 80–90 слов вместе со строкой источников.
ANSWER_MAX_CHARS = int(_env("ANSWER_MAX_CHARS", default="400"))

TEMPERATURE = float(_env("TEMPERATURE", default="0"))
MAX_TOKENS = int(_env("MAX_TOKENS", default="900"))

# --- лимиты времени одного прогона -----------------------------------------
# Регламент: 20 минут на весь прогон решения.
TIME_BUDGET_SEC = int(_env("TIME_BUDGET_SEC", default=str(20 * 60)))
# Сколько вопросов обрабатывать параллельно. 4 — компромисс: прогон 200 вопросов
# укладывается в бюджет, но не упираемся в лимиты провайдера по частоте запросов.
SOLVER_WORKERS = int(_env("MAX_WORKERS", "SOLVER_WORKERS", default="4"))
TIME_SAFE_SEC = int(_env("TIME_SAFE_SEC", default=str(17 * 60)))

# Имена файлов, под которыми может прийти база знаний
KB_ZIP_NAMES = ("knowledge-base.zip", "knowledge_base.zip", "kb.zip")


def describe() -> str:
    """Что настроено — для отладки. Ключи НИКОГДА не печатаем, только факт наличия."""
    def mask(value: str) -> str:
        return "задан" if value else "НЕ ЗАДАН"

    return "\n".join([
        f"генерация:    {LLM_BASE_URL or '(пусто)'} | модель={LLM_MODEL or '(пусто)'} | ключ={mask(LLM_API_KEY)}",
        f"эмбеддинги:   {EMB_BASE_URL or '(пусто)'} | модель={EMB_MODEL} | ключ={mask(EMB_API_KEY)}",
        f"реранк:       режим={RERANK_MODE} | модель={RERANK_MODEL or '(не задана)'} | ключ={mask(RERANK_API_KEY)}",
        f"поиск:        BM25 top-{BM25_TOP_K} | векторы top-{VECTOR_TOP_K} | RRF_K={RRF_K}",
        f"слияние:      вес BM25={BM25_WEIGHT} | глубина BM25={BM25_FUSION_DEPTH} | префикс запроса={'есть' if EMB_QUERY_PREFIX else 'нет'}",
    ])
