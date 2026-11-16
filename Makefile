PYTHON ?= python3
COMPOSE ?= docker compose
DEV = $(COMPOSE) -f compose.yaml -f compose.dev.yaml
PROD = $(COMPOSE) -f compose.yaml -f compose.prod.yaml

.PHONY: help dev dev-down dev-logs prod prod-down prod-logs build config test web submit benchmark chunks index reindex demo platform-zip platform-test docker
help:
	@printf '%s\n' 'make dev / dev-down / dev-logs  — разработка и Dozzle' 'make prod / prod-down / prod-logs — production' 'make build / config / test / submit — сборка, проверка, тесты, ZIP' 'make web / benchmark / demo — локальный запуск и оценка поиска'
dev:
	$(DEV) up -d --build --remove-orphans
dev-down:
	$(DEV) down
dev-logs:
	$(DEV) logs -f
prod:
	$(PROD) up -d --build --remove-orphans
prod-down:
	$(PROD) down
prod-logs:
	$(PROD) logs -f
build:
	$(PROD) build
config:
	$(DEV) config --quiet
	$(PROD) config --quiet
chunks:
	$(PYTHON) -m rag_pipeline.indexing --chunks-only
index:
	$(PYTHON) -m rag_pipeline.indexing
reindex:
	$(PYTHON) -m rag_pipeline.indexing --force
submit:
	$(PYTHON) tools/build_platform_submission.py
benchmark:
	$(PYTHON) tools/benchmark_retrieval.py
demo:
	$(PYTHON) solution.py --questions data/demo_questions.csv --out artifacts/demo/submission.csv --trace artifacts/demo/answer_trace.jsonl
test:
	PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v
	PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) platform_solution/tests/test_solution.py
	PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:query_processing_module $(PYTHON) -m unittest discover -s query_processing_module/tests -v
web:
	$(PYTHON) -m uvicorn API.app:app --host 0.0.0.0 --port 8000
docker: dev
platform-zip:
	$(PYTHON) platform_solution/build_zip.py
platform-test:
	PYTHONUTF8=1 $(PYTHON) platform_solution/tests/test_solution.py
