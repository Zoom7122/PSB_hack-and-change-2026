# Query Processing

Сервис подготавливает юридический вопрос к поиску: восстанавливает контекст диалога, определяет статус, извлекает реквизиты и создаёт 1–3 поисковые формулировки. Он не отвечает на вопрос и не проверяет наличие ответа в базе. Качество классификации, контекста и извлечения зависит от выбранной LLM; историю ответов ассистента нельзя считать источником права.

## Настройка

Нужен Python 3.10 или новее и доступ к OpenAI-совместимому Chat Completions API. Скопируйте `.env.example` в `.env` и задайте:

- `LLM_BASE_URL`: адрес API, например `https://openrouter.ai/api/v1` или `https://api.groq.com/openai/v1`.
- `LLM_API_KEY`: ключ поставщика; не добавляйте `.env` в Git.
- `LLM_MODEL`: идентификатор модели у выбранного поставщика.
- `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`, `LLM_TIMEOUT`: настройки генерации и таймаут в секундах. Если поставщик не поддерживает `temperature` или `max_tokens`, оставьте соответствующее значение пустым, чтобы не отправлять параметр.
- `QUERY_PROCESSING_MODE`: `single` (по умолчанию) или `multi`; выбирает реализацию для HTTP.
- `HOST`, `PORT`: адрес и порт HTTP-сервера.

При локальном запуске переменные процесса имеют приоритет над `.env`. Приложение проверяет конфигурацию при старте. Если модель дважды вернула неправильный JSON или нарушила схему, сервис возвращает ошибку; вызовы к провайдеру при сетевой ошибке автоматически не повторяются.

## Режимы обработки

`single` выполняет все действия в одном вызове LLM. `multi` выполняет отдельные этапы:

1. Восстанавливает смысл вопроса по истории или возвращает один уточняющий вопрос (`clarification_required`).
2. Отдельным вызовом определяет юридическую или нормативную тематику; непрофильный вопрос получает `out_of_scope`.
3. Для профильного вопроса формирует поисковые запросы и извлекает реквизиты (`ready`). Наличие данных в базе не проверяется.

В режиме `multi` выполняется от одного до трёх вызовов в зависимости от результата, плюс максимум один повтор на весь запрос при неверном JSON или несоответствии схеме. Повторные вызовы могут увеличить задержку и стоимость. Оба режима используют один формат входа и выхода. Ошибки API и таймауты автоматически не повторяются.

Чтобы переключить HTTP-сервис, задайте `QUERY_PROCESSING_MODE=multi` в `.env` и пересоздайте контейнер командой `docker compose up -d`. Прямой вызов многошагового сервиса:

```python
from query_processing.multi_stage import MultiStageQueryProcessingService

service = MultiStageQueryProcessingService(client)  # client — OpenAIClient или совместимая подмена
result = await service.process_query({"question": "А как его расторгнуть?", "history": []})
```

Качество каждого этапа зависит от модели: тесты с подменой ответов не подтверждают правильность её юридической классификации или постановки уточняющих вопросов.

## Docker

```bash
cp .env.example .env
# Заполните LLM_BASE_URL, LLM_API_KEY и LLM_MODEL в .env.
docker compose up --build -d
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query/process \
  -H 'Content-Type: application/json' \
  -d '{"question":"Как расторгнуть договор?","history":[]}'
docker compose logs -f
docker compose down
```

По умолчанию опубликован один порт; при масштабировании экземпляров настройте внешний балансировщик и публикацию портов в отдельном Compose override. Контейнер не хранит состояние запроса. `/health` показывает доступность приложения, но не проверяет работоспособность LLM.

## Локально и прямой вызов

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Для платформы организаторов вызовите тот же сервис напрямую; ниже пример использования настроек из окружения без привязки к интерфейсу будущего `solution.py`:

```python
import asyncio

from query_processing.config import LLMSettings
from query_processing.llm_client import OpenAIClient
from query_processing.service import QueryProcessingService


async def main():
    client = OpenAIClient(LLMSettings.from_environment())
    try:
        service = QueryProcessingService(client)
        result = await service.process_query({"question": "Как расторгнуть договор?", "history": []})
        print(result)
    finally:
        await client.close()


asyncio.run(main())
```

Можно передать собственный клиент с асинхронным методом `complete(messages) -> str`, возвращающим JSON модели. Метод `process_query` принимает словарь или `QueryRequest`, возвращает словарь и выбрасывает `pydantic.ValidationError` для неверного входа, `InvalidLLMResponse` для неверного ответа модели, `LLMTimeoutError` или `LLMError` для ошибок встроенного клиента.

## HTTP-контракт

`POST /query/process` принимает `question` и необязательную `history` — массив объектов `{"role": "user" | "assistant", "content": "..."}`. Ответ содержит `status`, `search_request` и `message`: при `ready` задан `search_request`, при `clarification_required` и `out_of_scope` он равен `null`, а `message` содержит пояснение. При ошибке входа HTTP 422, ответа/доступа к LLM — 502, при таймауте — 504. Поля фильтра, которые не удалось определить, равны `null`; дата имеет формат `YYYY-MM-DD`.

## Тесты

```bash
python -m unittest discover -s tests -v
```

Тесты подменяют ответы LLM; реальный провайдер и база документов не используются. Фикстуры проверяют контракт и передачу контекста, но не доказывают качество извлечения или классификации конкретной моделью.
