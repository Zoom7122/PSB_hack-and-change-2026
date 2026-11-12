# Модуль 5: Answer Generation + Verification

Формирует ответ юристу **только** по фрагментам, пришедшим от Reranker, и проверяет каждое юридически значимое утверждение до отправки пользователю. Если подтверждения нет, модуль возвращает обоснованный отказ (`insufficient`) и объясняет, каких сведений не хватает.

Контракт: `GenerationRequest → GenerationResult`, [CONTRACTS-ALL.md](../CONTRACTS-ALL.md), §5. Выход строго по контракту, вход терпим к расширениям модуля 4.

## Конвейер

```
GenerationRequest
  │
  ├─ 1. Подготовка доказательств (без LLM)            evidence.py
  │     • дедуп по chunk_id, сверка corpus_version
  │     • вырезание prompt-injection из текста документов
  │     • дата события: event_date или из вопроса («в августе 2024 года»)
  │     • применимость редакции каждого документа на эту дату
  │       (effective_from/to, а если их нет — «утратила силу с…», «действует с…»,
  │        «Статья 70 изменена с 1 сентября 2026 г.» в законах)
  │     • сверка чисел из вопроса с документами (ловушка «почему 500 000?»)
  │
  ├─ 2. Гейт: нет фрагментов / слабые scores поиска → insufficient без вызова LLM
  │
  ├─ 3. Генерация (LLM #1): JSON из предложений, у каждого kind и номера фрагментов
  │        fact — что сказано в документе · conclusion — вывод · gap — чего нет в базе
  │
  ├─ 4. Проверка                                        verification.py
  │     • программно: номер фрагмента существует; каждое число/дата/статья
  │       есть в тексте источника (450 000 = «450000»; 6 = «шести»; 12 млн = 12 000 000)
  │     • LLM #2: подтверждает ли текст утверждение, применима ли редакция,
  │       верна ли фраза «сведений нет» (защита от ложных отказов); исправляет
  │       привязку к источникам
  │
  ├─ 5. Исправление (только при замечаниях): черновик + замечания → LLM → проверка заново
  │
  └─ 6. Сборка: в ответ попадают только подтверждённые предложения
        → answered (verification=passed) | insufficient (+ что не хватает)
```

Число, которого нет ни в источнике, ни в вопросе, отсекается программно, даже если LLM-проверяющий счёл утверждение верным. Так ловится, например, «90 дней» из инъекции в служебной записке.

**Вызовы LLM:** обычно 2 (генерация + проверка), при замечаниях 4. В режиме `ANSWER_VERIFICATION_MODE=rules` всегда 1. Сбой модели не приводит к исключению: возвращается валидный `insufficient` с причиной в `verification.notes`.

## Как устроены ловушки базы и что с ними делает модуль

| Ловушка (из кейса/базы) | Механизм |
|---|---|
| Выдуманная норма/число | ответ из предложений с источниками + программная сверка чисел + LLM-проверка |
| Ложная посылка («почему 500 000?») | сверка чисел вопроса с фрагментами → подсказка в промпте, правило 4 |
| Редакция на дату события (регламент 2023/2025) | `applicability` по датам → модель видит «ещё не действовала» / «утратила силу» |
| Закон только в действующей редакции | маркеры «изменена с …» → пометка «редакции на дату события в базе нет» |
| Инъекция в документе (служебная записка ЮО-7) | предложения с инструкциями для ИИ вырезаются до промпта; документы в тегах как данные |
| Давление/поддельные системные сообщения в вопросе | вопрос в `<question>` как данные, экранирование тегов, правило 10 |
| Ложный отказ («в базе нет», хотя есть) | проверка gap-фраз LLM (`gap_wrong`) → исправление |
| «Альфа-Трейд» vs «Альфа-Трейдинг» | правило 5 + числа сверяются только с источниками того же документа |
| Противоправный запрос | правило 9: требования и последствия по базе, без инструкций по обходу |

## Интеграция

### HTTP (модуль 1 → модуль 5)

```bash
curl -X POST http://localhost:8005/answer/generate -H 'Content-Type: application/json' \
     -d @examples/rent_question.json
# ?debug=true — добавить поле debug: фрагменты, применимость, вердикты, тайминги
```

### В процессе (async, как модуль 2)

```python
from answer_generation import AnswerGenerationService, OpenAIClient, LLMSettings
from answer_generation.adapters import build_generation_request, sources_for_response

service = AnswerGenerationService(OpenAIClient(LLMSettings.from_environment()))
request = build_generation_request(rerank_result, query_result, request_id=request_id)  # модули 4 + 2
result = await service.generate(request)                                            # GenerationResult
chat_sources = sources_for_response(result, rerank_result)  # SourceReference[] для ChatResponse
```

Клиент модуля 2 (`OpenAIClient` с `async complete(messages)`) подходит как есть, отдельный создавать не нужно.

### Синхронно (solution.py, потоки)

```python
from answer_generation import generate_answer_sync, SyncLLMAdapter
result = generate_answer_sync(request, SyncLLMAdapter(llm))  # llm с методом chat(messages, temperature=)
```

`SyncLLMAdapter` принимает клиент из ветки `main` (`chat(...)`), любой объект с синхронным `complete(...)` или функцию `fn(messages) -> str`. Безопасно в `ThreadPoolExecutor` (покрыто тестом).

Для платформы скопируйте папку `answer_generation/` в корень архива. Зависимости ядра — `pydantic`, `python-dotenv` (оба есть в разрешённом списке); `openai` нужен только для `OpenAIClient`.

### Маппинг полей

| Откуда | Поле | Как используется |
|---|---|---|
| модуль 2 | `search_request.standalone_query` | `question` (через `build_generation_request`) |
| модуль 2 | `filters.event_date` | `event_date` |
| модуль 4 | `chunks[]` целиком | `chunks`; поля сверх контракта сохраняются |
| модуль 4 | `cite_label` (если есть) | человекочитаемая ссылка в промпте |
| модуль 4 | `scores.bm25/vector` (если есть у **всех** чанков) | гейт отказа до LLM |
| модуль 5 → 1 | `citations[].chunk_id` | `sources_for_response` берёт реквизиты из чанков, не от LLM |

## Отличия от буквы контракта (и почему)

- **Вход терпимый:** `revision_id`, `page`, `source_path`, `effective_*` могут быть `null` или мусором (станут `null`), лишние поля разрешены. Модуль 4 сейчас так и отдаёт (`tests/data/module4_rerank_result.json` из ветки data-processing).
- **`invalid_chunk_ids`** при `failed` содержит `#N`: это номер фрагмента в промпте, на который модель сослалась, хотя его не было. Модель не видит `chunk_id` и не может их выдумать, поэтому настоящие несуществующие `chunk_id` в ответе появиться не могут. Чанки чужой `corpus_version` во вход генерации не попадают, они перечислены в `notes`.
- **`status=insufficient` с непустыми `claims`** — частичный ответ: подтверждённые факты плюс явное указание, чего не хватает. `verification.status=passed`.
- **При `answered`** в `claims` только `supported` (контракт). Что было исключено и почему, видно в `verification.notes`.

## Запуск

```bash
cp .env.example .env            # заполнить LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
docker compose up --build -d    # http://localhost:8005, GET /health
```

Локально:

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --port 8005
python -m answer_generation examples/*.json --debug   # прогон примеров на настоящей модели
```

Порт 8005 выбран, чтобы не конфликтовать с модулями 1 и 2 (8000).

## Переменные окружения

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | — | OpenAI-совместимый API (те же, что у модуля 2) |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` | 0 / 1500 / 60 | параметры вызова |
| `LLM_JSON_MODE` | false | `response_format=json_object`; если провайдер его не принимает, отключается автоматически |
| `ANSWER_MAX_WORDS` | 150 | лимит слов (регламент платформы) |
| `ANSWER_MAX_CONTEXT_CHUNKS` / `ANSWER_MAX_CHUNK_CHARS` | 8 / 3000 | сколько контекста уходит в модель |
| `ANSWER_VERIFICATION_MODE` | llm | `llm` — правила + LLM-проверка; `rules` — только правила (быстрее, 1 вызов) |
| `ANSWER_MAX_REPAIRS` | 1 | циклов «замечания → исправление» |
| `ANSWER_LLM_RETRIES` | 1 | повторов при сетевой ошибке/таймауте |
| `ANSWER_EVIDENCE_GATE`, `ANSWER_GATE_BM25_MIN`, `ANSWER_GATE_VECTOR_MIN` | true, 4.0, 0.30 | отказ до LLM по слабым `scores` (пороги из замеров модулей 3–4) |
| `ANSWER_INFER_EVENT_DATE` | true | извлекать дату события из вопроса, если `event_date=null` |

## Тесты

```bash
python -m unittest discover -s tests -t . -v
```

50 тестов без сети: контракт и его инварианты, гейты, ложные посылки, выдуманные числа, исправление ссылок, ложные отказы, инъекции, редакции на дату, сбои LLM, неверный JSON, лимит слов, HTTP, синхронный адаптер в пуле потоков, стыковка на настоящих фикстурах модулей 2 и 4.

## Ограничения

- Качество смысловой проверки зависит от модели. Программная сверка чисел страхует только факты с числами.
- Номера частей и абзацев («ч. 3 ст. 70») жёстко не проверяются: в выгрузке законов части не пронумерованы. Номера статей и пунктов с точкой («ст. 70», «п. 3.1») проверяются.
- Применимость редакций по тексту работает на формулировках этой базы. Если Indexing начнёт заполнять `effective_from/to`, они имеют приоритет.
- Модуль отвечает по тому, что ему передали. Если Reranker не принёс нужный фрагмент, честный результат — `insufficient`.
