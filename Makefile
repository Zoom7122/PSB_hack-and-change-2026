PYTHON ?= python3
.PHONY: chunks index reindex verify

chunks:
	$(PYTHON) -m rag_pipeline.indexing --chunks-only

index:
	$(PYTHON) -m rag_pipeline.indexing

reindex:
	$(PYTHON) -m rag_pipeline.indexing --force

verify:
	$(PYTHON) -m answer_generation.verify
