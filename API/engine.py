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
from API.fallback import recover
from data_processing.textproc import evidence_text

LOG = logging.getLogger("legal_assistant")
ROOT = Path(__file__).resolve().parent.parent
MAX_HISTORY_MESSAGES = 8          # последние реплики диалога, передаваемые модели
EXCERPTS_PER_DOCUMENT = 2

ANSWERED_MODES = {"llm", "llm-fast", "llm-repaired", "modular-recovery"}
RECOVERY_MODES = {"error", "no-context", "extractive-llm-error"}


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
                 transport: httpx.BaseTransport | None = None, background_index: bool = True,
                 deadline: float = float("inf")) -> None:
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
        if self.api:
            self.api.deadline = deadline
        self.retriever = engine.Retriever(self.kb, self.api, self.settings)
        self.assistant = engine.Assistant(self.kb, self.retriever, self.api, self.settings, deadline)
        self.index_state = "disabled" if not (self.api and self.settings.use_embeddings) else "building"
        LOG.info("База знаний: %d документов, %d фрагментов; LLM %s", len(self.kb.documents),
                 len(self.kb.chunks), "подключена" if configured else "не настроена")
        if self.index_state == "building":
            if background_index:
                threading.Thread(target=self._build_index, daemon=True).start()
            else:
                self._build_index()

    def _build_index(self) -> None:
        """Строит платформенный векторный индекс или оставляет поиск BM25."""
        self.retriever.build_vectors([], min(self.assistant.deadline, time.monotonic() + 180))
        self.index_state = "ready" if self.retriever.matrix is not None else "bm25-only"

    def close(self) -> None:
        """Закрывает подключения к платформенному прокси."""
        if self.api is not None:
            self.api.close()

    # -- ответы ---------------------------------------------------------------
    def answer(self, parsed: engine.Question, pending: int = 0) -> engine.Outcome:
        """Выполняет платформенный путь и восстанавливает ответ при его сбое."""
        self.retriever.embed_query(parsed)
        try:
            outcome = self.assistant.answer(parsed, pending)
        finally:
            self.retriever.query_vectors.pop(parsed.qid, None)
        if outcome.mode in RECOVERY_MODES and self.api:
            result = recover(parsed.text, self.settings.data_dir, self.assistant.deadline)
            if result:
                return self._recovered_outcome(parsed, outcome, result)
        return outcome

    def _recovered_outcome(self, parsed: engine.Question, original: engine.Outcome, result: dict) -> engine.Outcome:
        """Сопоставляет проверенные источники резерва с документами веб-каталога."""
        cited = {cid for claim in result.get("claims", []) for cid in claim.get("chunk_ids", [])}
        by_name = {}
        for doc in self.kb.documents:
            by_name.setdefault((doc.group, doc.name), []).append(doc)
        sources = []
        for source in result["sources"]:
            matches = by_name.get(engine._split_name(source["source_path"]), [])
            if source["chunk_id"] not in cited or len(matches) != 1:
                continue
            doc = matches[0]
            excerpt = re.sub(r"\s+", " ", evidence_text(source["text"])).strip()
            sources.append({"chunk_id": source["chunk_id"], "document_id": str(doc.idx),
                            "document_title": doc.title, "revision_id": doc.status,
                            "section": source.get("section"), "page": None,
                            "source_path": doc.name, "kind": doc.kind,
                            "outdated": "УТРАТИЛ СИЛУ" in doc.status, "excerpt": excerpt[:900]})
        if len(sources) != len(cited):
            return original
        outcome = engine.Outcome(parsed.qid, result["answer"], "modular-recovery",
                                 issues=original.issues, seconds=original.seconds,
                                 doc_ids=list(dict.fromkeys(int(s["document_id"]) for s in sources)))
        outcome.recovery_sources = sources
        return outcome

    def ask(self, question: str, history: list[dict]) -> dict:
        """Возвращает ответ и источники для HTTP-интерфейса."""
        qid = "web-" + uuid.uuid4().hex[:10]
        parsed = engine.Question.parse(qid, dialog_text(question, history))
        outcome = self.answer(parsed)
        status = "answered" if outcome.mode in ANSWERED_MODES else "insufficient"
        return {"status": status, "answer": outcome.answer, "mode": outcome.mode, "seconds": outcome.seconds,
                "sources": self._sources(outcome)}

    def _sources(self, outcome: engine.Outcome) -> list[dict]:
        """Собирает ссылки на документы, использованные при ответе."""
        if hasattr(outcome, "recovery_sources"):
            return outcome.recovery_sources
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
        """Преобразует фрагмент платформенного корпуса в формат интерфейса."""
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
