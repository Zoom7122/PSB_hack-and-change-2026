# Юридический RAG: платформенный запуск

`solution.py` читает `DATA_DIR/questions.csv` с колонками `id,question` и архив
`DATA_DIR/knowledge-base.zip`. Запуск создаёт `submission.csv` в текущем
каталоге и `ARTIFACTS_DIR/answer_trace.jsonl` с фрагментами и проверкой ответов.
Запросы к поиску и построение индекса выполняют существующие компоненты команды.

```bash
python3 -m venv .venv-local
. .venv-local/bin/activate
python -m pip install -r requirements.txt
cp -n .env.example .env
# Задайте параметры LLM и эмбеддингов в .env.
python solution.py
```

Локальный пример вопросов находится в `data/demo_questions.csv`; для запуска
его можно скопировать в `data/questions.csv`. Платформа предоставляет свой CSV.

```bash
make chunks                 # ZIP -> artifacts/chunks.jsonl, без сети
make index                  # построить или загрузить индекс эмбеддингов
make reindex                # принудительно пересчитать индекс
make verify                 # повторно проверить сохранённые ответы
```

Для другого интерпретатора используйте, например, `make index PYTHON=.venv-local/bin/python`.
`DATA_DIR` и `ARTIFACTS_DIR` действуют и для Makefile, и для `solution.py`.
Векторный индекс сверяется с версией корпуса и названием модели. При запуске
`solution.py` отсутствие доступа к эмбеддингам переводит поиск на BM25; команды
`make index` и `make reindex` сообщают об ошибке, если индекс построить нельзя.

`QUERY_PROCESSING_MODE` выбирает `single` или `multi`, `MAX_WORKERS` управляет
параллельностью. `ANSWER_VERIFY_MAX_RETRIES=2` означает одну первоначальную и
до двух повторных генераций. Если ссылки или смысл не удаётся подтвердить, в CSV
попадает отказ с объяснением недостаточности подтверждения.

Проверка является частью модуля генерации и выполняется до записи CSV. Её можно
вызвать повторно отдельно:

```bash
python -m answer_generation.verify --submission submission.csv \
  --trace artifacts/answer_trace.jsonl --out submission.verified.csv \
  --out-trace artifacts/answer_trace.verified.jsonl
```

Отдельный запуск заново проверяет ссылки и смысл, а при ошибках перегенерирует
ответ по актуальным фрагментам. По умолчанию он пишет исправленные копии и не
перезаписывает исходные файлы. Проверка LLM является оценкой поддержки, а не
гарантией юридической правильности. Связи документов учитываются в том объёме,
в котором их предоставляют поисковые модули команды.
