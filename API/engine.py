"""Движок веб-чата: тот же поиск и генерация, что и в платформенном решении.

Используется `platform_solution/solution.py` — именно этот код проходит проверку
LLM-судьёй, поэтому веб-интерфейс отвечает так же, как сабмит. База знаний
загружается один раз при старте, векторный индекс строится в фоне: пока он не готов,
поиск работает на BM25.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import httpx

from platform_solution import solution as engine

LOG = logging.getLogger("legal_assistant")
ROOT = Path(__file__).resolve().parent.parent
MAX_HISTORY_MESSAGES = 8          # последние реплики диалога, передаваемые модели
EXCERPTS_PER_DOCUMENT = 2

ANSWERED_MODES = {"llm", "llm-fast", "llm-repaired"}


def load_env_file(path: Path = ROOT / ".env") -> None:
    """Подхватывает .env без перезаписи уже заданных переменных окружения."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if key.strip() and value:
            os.environ.setdefault(key.strip(), value)


def dialog_text(question: str, history: list[dict]) -> str:
    """Диалог в формате проверочных вопросов: «Пользователь: … / Ассистент: …»."""
    turns = [m for m in history if m.get("content", "").strip()][-MAX_HISTORY_MESSAGES:]
    if not turns:
        return question.strip()
    lines = [("Пользователь: " if m["role"] == "user" else "Ассистент: ") + m["content"].strip() for m in turns]
    lines.append("Пользователь: " + question.strip())
    return "\n".join(lines)


class LegalAssistant:
    def __init__(self, settings: engine.Settings | None = None,
                 transport: httpx.BaseTransport | None = None, background_index: bool = True) -> None:
        settings = settings or engine.Settings.from_env()
        data_dir = settings.data_dir if settings.data_dir.is_absolute() else ROOT / settings.data_dir
        reranker = settings.reranker_model or os.environ.get("RERANK_MODEL", "").strip()
        self.settings = dataclasses.replace(settings, data_dir=data_dir, trace=None, reranker_model=reranker)
        kb_path = engine.find_file(data_dir, "knowledge-base.zip", "kb.zip", "knowledge_base.zip", "knowledge-base")
        if kb_path is None:
            raise FileNotFoundError(f"Не найдена база знаний в {data_dir}")
        self.kb = engine.KnowledgeBase.from_sources(engine.read_sources(kb_path))
        configured = bool(self.settings.base_url and self.settings.api_key)
        self.api = engine.Api(self.settings, transport) if configured else None
        self.retriever = engine.Retriever(self.kb, self.api, self.settings)
        self.assistant = engine.Assistant(self.kb, self.retriever, self.api, self.settings, float("inf"))
        self.index_state = "disabled" if not (self.api and self.settings.use_embeddings) else "building"
        LOG.info("База знаний: %d документов, %d фрагментов; LLM %s", len(self.kb.documents),
                 len(self.kb.chunks), "подключена" if configured else "не настроена")
        if self.index_state == "building":
            if background_index:
                threading.Thread(target=self._build_index, daemon=True).start()
            else:
                self._build_index()

    def _build_index(self) -> None:
        self.retriever.build_vectors([], time.monotonic() + 300)
        self.index_state = "ready" if self.retriever.matrix is not None else "bm25-only"

    def close(self) -> None:
        if self.api is not None:
            self.api.close()

    # -- ответы ---------------------------------------------------------------
    def ask(self, question: str, history: list[dict]) -> dict:
        qid = "web-" + uuid.uuid4().hex[:10]
        parsed = engine.Question.parse(qid, dialog_text(question, history))
        self.retriever.embed_query(parsed)
        try:
            outcome = self.assistant.answer(parsed)
        finally:
            self.retriever.query_vectors.pop(qid, None)
        if outcome.mode in ANSWERED_MODES:
            status = "answered"
        else:
            status = "insufficient"
        return {"status": status, "answer": outcome.answer, "mode": outcome.mode, "seconds": outcome.seconds,
                "sources": self._sources(outcome)}

    def _sources(self, outcome: engine.Outcome) -> list[dict]:
        rank = {chunk: r for r, chunk in enumerate(outcome.top_chunks)}
        sources = []
        for doc_id in outcome.doc_ids:
            doc = self.kb.documents[doc_id]
            best = sorted((c for c in doc.chunks if c in rank), key=rank.__getitem__)[:EXCERPTS_PER_DOCUMENT]
            if not best:
                best = doc.chunks[:1]
            for chunk_id in best:
                sources.append(self._source(doc, chunk_id))
        return sources

    def _source(self, doc: engine.Document, chunk_id: int) -> dict:
        excerpt = re.sub(r"\s+", " ", self.kb.chunks[chunk_id].text).strip()
        return {
            "chunk_id": f"{doc.idx}:{self.kb.chunks[chunk_id].pos}",
            "document_id": str(doc.idx),
            "document_title": doc.title,
            "revision_id": doc.status,
            "section": engine._law_citation(doc.title) if doc.kind == "law" else None,
            "page": None,
            "source_path": doc.name,
            "kind": doc.kind,
            "outdated": "УТРАТИЛ СИЛУ" in doc.status,
            "excerpt": excerpt[:900] + ("…" if len(excerpt) > 900 else ""),
        }

    # -- справочные данные ----------------------------------------------------
    def document(self, doc_id: int) -> dict | None:
        if not 0 <= doc_id < len(self.kb.documents):
            return None
        doc = self.kb.documents[doc_id]
        return {"document_id": str(doc.idx), "title": doc.title, "kind": doc.kind, "status": doc.status,
                "source_path": doc.name, "text": doc.text,
                "outdated": "УТРАТИЛ СИЛУ" in doc.status}

    def documents(self) -> list[dict]:
        return [{"document_id": str(d.idx), "title": d.title, "kind": d.kind, "status": d.status,
                 "outdated": "УТРАТИЛ СИЛУ" in d.status} for d in self.kb.documents]

    def health(self) -> dict:
        return {"documents": len(self.kb.documents), "chunks": len(self.kb.chunks),
                "llm": "configured" if self.api else "not_configured",
                "model": self.settings.llm_model if self.api else None,
                "vector_index": self.index_state,
                "reranker": "on" if self.api and self.api.rerank_available else "off"}
