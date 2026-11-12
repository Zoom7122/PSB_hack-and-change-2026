# Контракты модулей — юридический RAG-ассистент ПСБ

**Версия контрактов:** `1.0-draft`  
**Формат:** JSON по HTTP или при вызове функций внутри приложения; контракты не зависят от языка программирования.  
**Статус:** предложение для согласования командой, не официальный формат платформы хакатона.

## Состав модулей

| № | Модуль | Вход → выход |
|---|---|---|
| 1 | API Backend | `ChatRequest` → `ChatResponse` |
| 2 | Query Processing | `QueryProcessingInput` → `QueryProcessingResult` |
| 3 | Hybrid RAG (Vector + BM25 + RRF) | `RetrievalRequest` → `RetrievalResult` |
| 4 | Reranker | `RerankRequest` → `RerankResult` |
| 5 | Answer Generation + Verification | `GenerationRequest` → `GenerationResult` |
| 6 | Data Indexing | `IndexingRequest` → `IndexingResult` и `DocumentChunk` в хранилище |

Frontend — потребитель API, не отдельный Backend-модуль. Отдельные Evidence Sufficiency и Verification не выделяются: оценка достаточности, повторный поиск и верификация входят во внутреннюю логику модулей 3–5 и вызывающего пайплайна.

## Порядок вызовов

`Frontend → API → Query Processing → Hybrid RAG → Reranker → Answer Generation/Verification → API → Frontend`

`Indexing` выполняется **до запросов**, сохраняет тексты чанков, метаданные и индексы, которыми пользуется Hybrid RAG.

## Общие соглашения

- Все объекты передаются в JSON UTF-8. Имена ключей в `snake_case`. В датах — `YYYY-MM-DD`, во времени — ISO 8601 UTC.
- `request_id` создаётся API при обработке запроса и передаётся сквозным образом. Внешний `ChatRequest` не обязан его содержать.
- **Отсутствующее значение:** `null`. **Нет результатов:** пустой массив `[]`. Не подменять пустой список искусственным чанком.
- `corpus_version` — версия набора документов и индексов. Она должна совпадать в Hybrid RAG, Reranker и Generation.
- `chunk_id` уникален **в рамках `corpus_version`** и не изменяется между Search → Rerank → Generate → Verify. `document_id` идентифицирует документ, `revision_id` — его редакцию.
- `document_title`, `section`, `page`, `source_path`, даты и редакции берутся только из хранилища документов. LLM не может их выдумывать.
- **Для юриста важно:** нельзя считать наличие ссылки доказательством — каждое юридически значимое утверждение должно подтверждаться текстом фрагмента.
- Вопрос + история передаются в Query Processing, а в Retrieval используется сформированный самостоятельный вопрос. Прошлые ответы ассистента **не являются юридическим источником**.
- `score`, `rrf_score` и `rerank_score` — оценки ранжирования, **не вероятности правдивости**.
- Для безопасности `source_path` — внутренний адрес доступного документа, а не URL, сгенерированный LLM. Внешний URL просмотра собирает API из доверенных данных.
- Во всех примерах ниже **договор №123 и его пункты вымышлены**.

## Ветка завершения без поиска

Query Processing может вернуть `clarification_required` или `out_of_scope`. В таком случае API сразу формирует `ChatResponse` и **не вызывает** Retrieval, Reranker и Generation.

## Для параллельной разработки

Сначала зафиксируйте поля и допустимые статусы; после этого каждый модуль может разрабатываться на JSON-фикстурах соседнего модуля. Формат `solution.py` (для платформы) — внешний адаптер, а не контракт между этими модулями.


---

# 1. API Backend

Вход для Vue/другого Frontend, маршрутизация вызова бизнес-логики и сборка ответа с проверенными метаданными источников.

## Вход: `ChatRequest`

```json
{
  "question": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
  "history": [
    {
      "role": "user",
      "content": "Какие условия расторжения договора №123?"
    },
    {
      "role": "assistant",
      "content": "По найденному договору может требоваться предварительное уведомление."
    }
  ]
}
```

## Выход: `ChatResponse`

```json
{
  "request_id": "req-001",
  "status": "answered",
  "answer": "Пункт 4.2 договора №123 устанавливает уведомление не менее чем за 30 дней. Пункт 4.3 допускает досрочное расторжение при существенном нарушении, но в найденных фрагментах не сказано, что уведомление можно не направлять. Поэтому подтвердить расторжение без уведомления на основании этих положений нельзя.",
  "sources": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "revision_id": "rev-2025",
      "section": "4.2",
      "page": 5,
      "source_path": "documents/contract_123.pdf"
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "revision_id": "rev-2025",
      "section": "4.3",
      "page": 5,
      "source_path": "documents/contract_123.pdf"
    }
  ]
}
```

## Обязательные правила контракта

- API делает валидацию входа, простую безопасную нормализацию и, при желании, точный кеш. Исходный вопрос сохраняется.
- При `out_of_scope`, `clarification_required` или `insufficient` допустимы `sources: []`.
- API не доверяет реквизитам источников от LLM: заполняет `sources` только из выбранных ранее чанков / хранилища.
- Кеширование ответа допускается только при равенстве вопроса, релевантной истории/контекста и версии базы. Semantic cache для юридических ответов по умолчанию отключён.

## Типы и ограничения

- `ChatRequest.question: string, обязательный, непустой`
- `ChatRequest.history: Message[], необязательный (по умолчанию [])`
- `Message.role: "user" | "assistant"`
- `Message.content: string`
- `ChatResponse.request_id: string`
- `ChatResponse.status: "answered" | "clarification_required" | "out_of_scope" | "insufficient"`
- `ChatResponse.answer: string`
- `ChatResponse.sources: SourceReference[]; пустой массив допустим`

### Пример: нужен контекст

```json
{
  "request_id": "req-002",
  "status": "clarification_required",
  "answer": "Уточните, к какому договору относится вопрос «А можно иначе?».",
  "sources": []
}
```


---

# 2. Query Processing

Один этап анализа вопроса: восстановление смысла по истории, классификация тематики, проверка ясности, извлечение реквизитов и формирование поисковых строк.

## Вход: `QueryProcessingInput`

```json
{
  "request_id": "req-001",
  "question": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
  "history": [
    {
      "role": "user",
      "content": "Какие условия расторжения договора №123?"
    },
    {
      "role": "assistant",
      "content": "По найденному договору может требоваться предварительное уведомление."
    }
  ]
}
```

## Выход: `QueryProcessingResult`

```json
{
  "request_id": "req-001",
  "status": "ready",
  "message": null,
  "search_request": {
    "standalone_query": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
    "search_queries": [
      "условия расторжения договора №123 без уведомления",
      "односторонний отказ от договора №123",
      "срок уведомления при расторжении договора №123"
    ],
    "filters": {
      "document_id": null,
      "document_number": "123",
      "document_type": "contract",
      "event_date": null
    }
  }
}
```

## Обязательные правила контракта

- Если `status="ready"`, `search_request` обязателен, `search_queries` не пустой.
- Если нужен контекст или вопрос непрофильный, `search_request=null`, а `message` объясняет причину.
- Не извлекать документ, дату, номер или редакцию, если пользователь их не назвал и они не следуют из истории.
- Не считать отсутствие номера договора автоматически причиной для уточнения: широкий юридический вопрос тоже можно искать.
- Не давать юридический ответ на этом этапе — только подготавливать поисковые запросы и фильтры.

## Типы и ограничения

- `status: "ready" | "clarification_required" | "out_of_scope"`
- `search_request: SearchRequest | null`
- `message: string | null`
- `search_request.standalone_query: string — смысл исходного вопроса с учётом истории`
- `search_request.search_queries: string[] — обычно 1–3 поисковые формулировки`
- `search_request.filters: объект, значения могут быть null`
- `filters.event_date: YYYY-MM-DD | null`

### Пример: недостаточно контекста

```json
{
  "request_id": "req-002",
  "status": "clarification_required",
  "message": "Уточните, о каком договоре или документе идёт речь.",
  "search_request": null
}
```

### Пример: непрофильный вопрос

```json
{
  "request_id": "req-003",
  "status": "out_of_scope",
  "message": "Ассистент отвечает только на вопросы по юридическим документам из базы знаний.",
  "search_request": null
}
```


---

# 3. Hybrid RAG (Vector + BM25 + RRF)

Принимает текстовые запросы и фильтры; ищет совпадения в уже подготовленных индексах; объединяет результаты с дедупликацией по `chunk_id`.

## Вход: `RetrievalRequest`

```json
{
  "request_id": "req-001",
  "search_request": {
    "standalone_query": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
    "search_queries": [
      "условия расторжения договора №123 без уведомления",
      "односторонний отказ от договора №123",
      "срок уведомления при расторжении договора №123"
    ],
    "filters": {
      "document_id": null,
      "document_number": "123",
      "document_type": "contract",
      "event_date": null
    }
  },
  "limits": {
    "vector_top_k": 20,
    "bm25_top_k": 20,
    "fused_top_k": 20
  }
}
```

## Выход: `RetrievalResult`

```json
{
  "request_id": "req-001",
  "corpus_version": "kb-v1",
  "candidates": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.2",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней.",
      "rrf_rank": 1,
      "rrf_score": 0.0325,
      "matched_by": [
        "vector",
        "bm25"
      ]
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.3",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Досрочное расторжение допускается в случае существенного нарушения условий договора.",
      "rrf_rank": 2,
      "rrf_score": 0.0161,
      "matched_by": [
        "vector"
      ]
    }
  ]
}
```

## Обязательные правила контракта

- Vector Search использует embedding-модель совместимую с индексом (в кейсе доступна BAAI/bge-m3).
- BM25 принимает текст/токены, а не SQL; RRF объединяет ранги двух списков и убирает дубли.
- При отсутствии результатов вернуть `candidates: []`, не выдумывать источник.
- Фильтры по точным реквизитам документа должны применяться согласованно; датированные ответы используют метаданные редакций.
- Каждый кандидат содержит исходный текст и метаданные из Indexing — их не меняет поисковый модуль.

## Типы и ограничения

- `search_request: структура Query Processing без изменений`
- `limits.vector_top_k / bm25_top_k / fused_top_k: integer > 0`
- `corpus_version: string`
- `candidates: RetrievedCandidate[]`
- `candidates[].rrf_rank: integer начиная с 1`
- `candidates[].rrf_score: number (не вероятность)`
- `candidates[].matched_by: массив из "vector" и/или "bm25"`

### Общий формат кандидата

```json
{
  "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
  "document_id": "contract-123",
  "document_title": "Договор №123",
  "document_type": "contract",
  "document_number": "123",
  "revision_id": "rev-2025",
  "effective_from": "2025-01-01",
  "effective_to": null,
  "section": "4.2",
  "page": 5,
  "source_path": "documents/contract_123.pdf",
  "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней.",
  "rrf_rank": 1,
  "rrf_score": 0.0325,
  "matched_by": [
    "vector",
    "bm25"
  ]
}
```


---

# 4. Reranker

Получает кандидатов от Hybrid RAG, пересортировывает их относительно исходного самостоятельного вопроса и выбирает лучшие чанки.

## Вход: `RerankRequest`

```json
{
  "request_id": "req-001",
  "corpus_version": "kb-v1",
  "question": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
  "candidates": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.2",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней.",
      "rrf_rank": 1,
      "rrf_score": 0.0325,
      "matched_by": [
        "vector",
        "bm25"
      ]
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.3",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Досрочное расторжение допускается в случае существенного нарушения условий договора.",
      "rrf_rank": 2,
      "rrf_score": 0.0161,
      "matched_by": [
        "vector"
      ]
    }
  ],
  "top_k": 8
}
```

## Выход: `RerankResult`

```json
{
  "request_id": "req-001",
  "corpus_version": "kb-v1",
  "chunks": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.2",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней.",
      "rank": 1,
      "rerank_score": 0.87
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.3",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Досрочное расторжение допускается в случае существенного нарушения условий договора.",
      "rank": 2,
      "rerank_score": 0.71
    }
  ]
}
```

## Обязательные правила контракта

- Не создаёт новые чанки: возвращает подмножество переданных кандидатов.
- Сохраняет `chunk_id`, `document_id`, текст, редакции и все прочие метаданные без изменений.
- При пустом входном массиве возвращает `chunks: []`.
- Модель Qwen3 Reranker 8B на платформе доступна, но этап можно временно заменить детерминированной заглушкой для интеграционных тестов.

## Типы и ограничения

- `question: string — исходный самостоятельный вопрос, не расширенный поисковый запрос`
- `candidates: RetrievedCandidate[]`
- `top_k: integer > 0`
- `chunks: RerankedChunk[]`
- `chunks[].rank: integer начиная с 1`
- `chunks[].rerank_score: number (не вероятность)`

### Требование к выходу

`chunks` сортируются по `rank` по возрастанию; каждый `chunk_id` встречается не более одного раза.


---

# 5. Answer Generation + Verification

Формирует обоснованный ответ по текстам из Reranker, проверяет корректность ссылок и подтверждение каждого юридически значимого утверждения. Проверка источников реализуется здесь же, без отдельного сервиса.

## Вход: `GenerationRequest`

```json
{
  "request_id": "req-001",
  "corpus_version": "kb-v1",
  "question": "Можно ли расторгнуть договор №123 без предварительного уведомления?",
  "event_date": null,
  "chunks": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.2",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней.",
      "rank": 1,
      "rerank_score": 0.87
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "document_id": "contract-123",
      "document_title": "Договор №123",
      "document_type": "contract",
      "document_number": "123",
      "revision_id": "rev-2025",
      "effective_from": "2025-01-01",
      "effective_to": null,
      "section": "4.3",
      "page": 5,
      "source_path": "documents/contract_123.pdf",
      "text": "Досрочное расторжение допускается в случае существенного нарушения условий договора.",
      "rank": 2,
      "rerank_score": 0.71
    }
  ]
}
```

## Выход: `GenerationResult`

```json
{
  "request_id": "req-001",
  "status": "answered",
  "answer": "Пункт 4.2 договора №123 устанавливает уведомление не менее чем за 30 дней. Пункт 4.3 допускает досрочное расторжение при существенном нарушении, но в найденных фрагментах не сказано, что уведомление можно не направлять. Поэтому подтвердить расторжение без уведомления на основании этих положений нельзя.",
  "claims": [
    {
      "claim_id": "claim-1",
      "text": "Договор №123 предусматривает уведомление о расторжении не менее чем за 30 дней.",
      "chunk_ids": [
        "kb-v1:contract-123:rev-2025:sec-4.2"
      ],
      "support": "supported"
    },
    {
      "claim_id": "claim-2",
      "text": "В найденных пунктах не установлена возможность расторжения без уведомления.",
      "chunk_ids": [
        "kb-v1:contract-123:rev-2025:sec-4.2",
        "kb-v1:contract-123:rev-2025:sec-4.3"
      ],
      "support": "supported"
    }
  ],
  "citations": [
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
      "claim_ids": [
        "claim-1",
        "claim-2"
      ]
    },
    {
      "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.3",
      "claim_ids": [
        "claim-2"
      ]
    }
  ],
  "verification": {
    "status": "passed",
    "invalid_chunk_ids": [],
    "unsupported_claim_ids": [],
    "notes": []
  }
}
```

## Обязательные правила контракта

- В LLM передавать **текст** чанков и метаданные, а не только ссылки или ID.
- Проверять программно, что каждый `chunk_id` существует среди **переданных** в `GenerationRequest.chunks` и соответствует `corpus_version`.
- Дополнительная проверка оценивает, действительно ли текст каждого чанка подтверждает привязанное утверждение (например, LLM-проверкой).
- Утверждения о действии нормы на дату события требуют проверки соответствующей редакции; при отсутствии необходимой редакции не делать выводов.
- При `status="answered"` допустим только `verification.status="passed"` и поддержанные claims.
- При нехватке фактов вернуть `status="insufficient"`, объяснить, какой информации не хватает; неподтверждённые ссылки не выдавать.
- При неудачной проверке допускается внутреннее исправление/повторное генерирование, но неподтверждённый ответ нельзя считать одобренным.

## Типы и ограничения

- `status: "answered" | "insufficient"`
- `claims: Claim[]`
- `claims[].claim_id: string`
- `claims[].text: string`
- `claims[].chunk_ids: string[]`
- `claims[].support: "supported" | "unsupported"`
- `citations: массив {chunk_id, claim_ids}`
- `verification.status: "passed" | "failed" | "not_applicable"`
- `verification.invalid_chunk_ids / unsupported_claim_ids: string[]`
- `event_date: YYYY-MM-DD | null`

### Пример: недостаточно доказательств

```json
{
  "request_id": "req-004",
  "status": "insufficient",
  "answer": "В предоставленных документах нет достаточного подтверждения возможности расторжения без уведомления для указанной ситуации.",
  "claims": [],
  "citations": [],
  "verification": {
    "status": "not_applicable",
    "invalid_chunk_ids": [],
    "unsupported_claim_ids": [],
    "notes": [
      "Не найдены применимые подтверждающие положения."
    ]
  }
}
```


---

# 6. Data Indexing

Независимый предварительный процесс. Извлекает текст и реквизиты из предоставленных документов, нарезает их на смысловые чанки, считает эмбеддинги, строит Vector Index + BM25 Index и сохраняет хранилище.

## Вход: `IndexingRequest`

```json
{
  "source_dir": "data/documents",
  "output_dir": "data/indexes",
  "force_rebuild": false
}
```

## Выход: `IndexingResult`

```json
{
  "status": "completed",
  "manifest": {
    "corpus_version": "kb-v1",
    "generated_at": "2026-11-10T18:00:00Z",
    "embedding_model": "BAAI/bge-m3",
    "vector_index_path": "data/indexes/vector.faiss",
    "bm25_index_path": "data/indexes/bm25",
    "chunk_store_path": "data/indexes/chunks.jsonl",
    "document_count": 12,
    "chunk_count": 438
  },
  "warnings": []
}
```

## Обязательные правила контракта

- Индексация не запускается заново при каждом вопросе. Готовые индексы подгружаются при старте Retrieval.
- Использовать одну и ту же embedding-модель и её параметры для индексации документов и векторизации поисковых запросов.
- Сохранять привязку чанка к названию документа, странице, пункту, `revision_id`, датам действия и исходному файлу.
- `chunk_id` стабилен в рамках версии корпуса. При изменении набора документов/нарезки обновляется `corpus_version`.
- Не указывать фиктивные даты действия — использовать `null`, если редакция/период неизвестны.

## Типы и ограничения

- `source_dir/output_dir: строковые пути или URI (договориться об общей файловой раскладке)`
- `force_rebuild: boolean`
- `status: "completed" | "failed"`
- `manifest.corpus_version: string`
- `manifest.embedding_model: string`
- `manifest.vector_index_path / bm25_index_path / chunk_store_path: string`
- `manifest.document_count / chunk_count: integer >= 0`
- `manifest.generated_at: строка ISO 8601 UTC`
- `warnings: string[]`

## Формат записи в хранилище: `DocumentChunk`

```json
{
  "chunk_id": "kb-v1:contract-123:rev-2025:sec-4.2",
  "document_id": "contract-123",
  "document_title": "Договор №123",
  "document_type": "contract",
  "document_number": "123",
  "revision_id": "rev-2025",
  "effective_from": "2025-01-01",
  "effective_to": null,
  "section": "4.2",
  "page": 5,
  "source_path": "documents/contract_123.pdf",
  "text": "Сторона обязана уведомить другую сторону о расторжении договора не менее чем за 30 календарных дней."
}
```

Эти поля также используются в `RetrievedCandidate` и `RerankedChunk`. Поиск и rerank только дополняют их оценками ранжирования.
