PYTHON ?= python3
.PHONY: chunks index reindex verify submit benchmark demo test

chunks:
	$(PYTHON) -m rag_pipeline.indexing --chunks-only

index:
	$(PYTHON) -m rag_pipeline.indexing

reindex:
	$(PYTHON) -m rag_pipeline.indexing --force

verify:
	$(PYTHON) -m answer_generation.verify

submit:
	$(PYTHON) tools/build_platform_submission.py

benchmark:
	$(PYTHON) tools/benchmark_retrieval.py

demo:
	$(PYTHON) solution.py --questions data/demo_questions.csv --out artifacts/demo/submission.csv --trace artifacts/demo/answer_trace.jsonl

test:
	PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests -v
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:query_processing_module $(PYTHON) -m unittest discover -s query_processing_module/tests -v
