"""Тонкий клиент к OpenAI-совместимым API: генерация, эмбеддинги, реранк.

Почему httpx, а не SDK openai
=============================
Ранее я предлагал `openai` для чата и эмбеддингов. При двух разных провайдерах
(DeepSeek для генерации, OpenAI для эмбеддингов, локальный прокси на платформе)
выгоднее один тонкий клиент на httpx:

  * один стиль ошибок и повторов на все три типа запросов;
  * реранк (`/rerank`) — не OpenAI-эндпоинт, его всё равно пришлось бы звать
    сырым HTTP, а значит в проекте было бы два разных способа работы с сетью;
  * нет риска, что версия SDK на платформе отличается от локальной.

httpx есть в разрешённом списке библиотек, так что этот выбор безопасен.

Три эндпоинта
=============
  POST {base}/chat/completions  — генерация (messages → текст)
  POST {base}/embeddings        — векторы (input: [текст, …] → [[число, …], …])
  POST {base}/rerank            — переранжирование (query + documents → порядок)

Обработка ошибок по регламенту
=============================
* 429, 5xx, таймауты — ограниченное число повторов с паузой (не бесконечный цикл);
* 401 — ключ неверный, повторять бессмысленно, сообщаем понятной ошибкой;
* любой сбой эмбеддингов или реранка НЕ должен ронять прогон: вызывающий код
  обязан уметь работать без них (это и есть graceful degradation).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import httpx

if __package__:
    from . import config
else:
    import config

# Таймаут на один запрос. 60 секунд — компромисс: генерация длинного ответа
# укладывается, а зависший запрос не съедает весь бюджет прогона.
DEFAULT_TIMEOUT = 60.0
MAX_RETRIES = 3
RETRY_PAUSE = 1.5      # секунды, умножается на номер попытки


class LLMError(RuntimeError):
    """Ошибка обращения к модели: сеть, авторизация, некорректный ответ."""


@dataclass
class LLMClient:
    """Клиент с тремя методами. Создаётся один раз и переиспользуется.

    Важно для многопоточности: httpx.Client безопасен для использования из
    нескольких потоков, поэтому в solution.py можно держать один клиент на весь
    прогон, а не создавать его на каждый вопрос.
    """

    timeout: float = DEFAULT_TIMEOUT
    deadline: float | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(timeout=self.timeout)

    # ---------------------------------------------------------------- служебное
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _headers(api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _post(self, url: str, api_key: str, payload: dict) -> dict:
        """POST с повторами. Возвращает JSON или бросает LLMError.

        Повторяем только то, что имеет смысл повторять: 429 (лимит), 5xx
        (временная ошибка сервера) и сетевые сбои. На 400/401/404 повтор
        ничего не изменит — сразу сообщаем ошибку.
        """
        if not url:
            raise LLMError("Не задан адрес API (LLM_BASE_URL / EMB_BASE_URL)")

        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            remaining = self.deadline - time.monotonic() if self.deadline else self.timeout
            if remaining <= 0:
                raise LLMError("Исчерпан бюджет времени запросов")
            try:
                response = self._client.post(url, headers=self._headers(api_key), json=payload,
                                             timeout=min(self.timeout, remaining))
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise LLMError(f"Ответ не является JSON: {url}: {exc}") from exc
                if response.status_code in (400, 401, 403, 404, 422):
                    raise LLMError(
                        f"API status={response.status_code}"
                    )
                last_error = f"HTTP {response.status_code}"

            if attempt < MAX_RETRIES:
                pause = RETRY_PAUSE * attempt
                if self.deadline:
                    pause = min(pause, max(0, self.deadline - time.monotonic()))
                time.sleep(pause)
        raise LLMError(f"Не удалось выполнить запрос {url} после {MAX_RETRIES} попыток: {last_error}")

    # ------------------------------------------------------------- генерация
    def chat(self, messages: list[dict], *, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
        """Сгенерировать ответ. Возвращает текст (никогда не None)."""
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
            "max_tokens": config.MAX_TOKENS if max_tokens is None else max_tokens,
        }
        data = self._post(f"{config.LLM_BASE_URL}/chat/completions", config.LLM_API_KEY, payload)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Неожиданная структура ответа chat: {str(data)[:300]}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMError("Модель вернула пустой ответ")
        return content.strip()

    # ----------------------------------------------------------- эмбеддинги
    def embed(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Посчитать векторы для списка текстов.

        Батчами, потому что отправлять 360 текстов одним запросом — риск упереться
        в лимит размера запроса, а по одному — 360 раз ждать сеть.

        Векторы НОРМИРУЮТСЯ к единичной длине: тогда косинусное сходство равно
        скалярному произведению, и поиск превращается в одно умножение матриц.
        """
        if not texts:
            return []
        size = batch_size or config.EMB_BATCH_SIZE
        vectors: list[list[float]] = []

        for start in range(0, len(texts), size):
            batch = texts[start:start + size]
            payload = {"model": config.EMB_MODEL, "input": batch, "encoding_format": "float"}
            data = self._post(f"{config.EMB_BASE_URL}/embeddings", config.EMB_API_KEY, payload)
            try:
                items = sorted(data["data"], key=lambda item: item.get("index", 0))
            except (KeyError, TypeError) as exc:
                raise LLMError(f"Неожиданная структура ответа embeddings: {str(data)[:300]}") from exc
            if len(items) != len(batch):
                raise LLMError(f"Пришло {len(items)} векторов на {len(batch)} текстов")
            vectors.extend(_normalize(item["embedding"]) for item in items)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Посчитать вектор ЗАПРОСА (с префиксом-инструкцией, если он задан).

        Почему отдельный метод: у асимметричного поиска запрос и документ
        кодируются по-разному. Модели семейства Qwen3-Embedding ожидают, что
        запрос будет оформлен как «Instruct: <задача>\nQuery: <текст>», а
        документы — как есть. Если кодировать запрос без инструкции, качество
        поиска падает (проверяется нашим бенчмарком).
        """
        prefix = config.EMB_QUERY_PREFIX
        return self.embed([f"{prefix}{text}" if prefix else text])[0]

    def embed_documents(self, texts: list[str], batch_size: int | None = None) -> list[list[float]]:
        """Посчитать векторы ДОКУМЕНТОВ — всегда без префикса."""
        return self.embed(texts, batch_size=batch_size)

    # ---------------------------------------------------------------- реранк
    def rerank(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Переранжировать документы через /rerank. Возвращает [(индекс, оценка), …]."""
        payload = {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        data = self._post(f"{config.RERANK_BASE_URL}/rerank", config.RERANK_API_KEY, payload)
        items = data.get("results") or data.get("data") or []
        result: list[tuple[int, float]] = []
        for item in items:
            try:
                index = int(item["index"])
                score = float(item.get("relevance_score", item.get("score", 0.0)))
            except (KeyError, TypeError, ValueError):
                continue
            if 0 <= index < len(documents):
                result.append((index, score))
        if not result:
            raise LLMError(f"Реранкер вернул пустой результат: {str(data)[:200]}")
        return result[:top_n]

    # ------------------------------------------------ реранк силами chat-модели
    def rerank_by_llm(self, query: str, documents: list[str], top_n: int) -> list[tuple[int, float]]:
        """Переранжировать документы силами обычной языковой модели.

        Зачем: у нас может не быть доступа к /rerank (локально его нет вообще),
        но есть chat-API. Тогда просим модель вернуть номера самых релевантных
        фрагментов. Это менее точно, чем специализированный реранкер, но
        заметно лучше, чем ничего, и одинаково работает с любой моделью.

        Формат ответа — только номера: «3, 1, 7». Любой другой текст считаем
        мусором и отдаём пустой список, чтобы вызывающий код не падал.
        """
        import re

        listing = "\n".join(f"[{i}] {doc[:1800]}" for i, doc in enumerate(documents))
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты помогаешь искать нормы в юридических документах. "
                    "Тебе дают вопрос и пронумерованные фрагменты. "
                    f"Верни номера {top_n} самых релевантных фрагментов по убыванию релевантности "
                    "в виде чисел через запятую, без пояснений и без текста. "
                    "Если ни один фрагмент не относится к вопросу, ответь: none"
                ),
            },
            {"role": "user", "content": f"Вопрос: {query}\n\nФрагменты:\n{listing}"},
        ]
        answer = self.chat(messages, temperature=0.0, max_tokens=64)
        if "none" in answer.lower():
            return []
        numbers = [int(n) for n in re.findall(r"\d+", answer)]
        seen: set[int] = set()
        ranked: list[tuple[int, float]] = []
        for position, number in enumerate(numbers):
            if number in seen or not (0 <= number < len(documents)):
                continue
            seen.add(number)
            # Оценку изображаем убывающей: специализированного скора у нас нет
            ranked.append((number, 1.0 - position * 0.01))
            if len(ranked) >= top_n:
                break
        return ranked


def _normalize(vector: list[float]) -> list[float]:
    """Привести вектор к единичной длине.

    Зачем: тогда косинус = скалярное произведение, и весь поиск — одно
    матричное умножение. Плюс разные модели дают векторы разной длины, а
    сравнивать их между собой нужно одинаково.
    """
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def make_client() -> LLMClient:
    """Создать клиент по текущим настройкам."""
    return LLMClient()
