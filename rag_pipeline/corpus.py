"""Загрузка корпуса и доступ к подлинным фрагментам для проверки ссылок."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

from data_processing import chunking, ingest, refs, search  # noqa: E402
from .editions import document_period, iso_date


def _citation(meta: chunking.DocMeta, chunk: chunking.Chunk) -> str:
    """Создаёт ссылку только из метаданных исходного документа."""
    title = meta.citation_title.strip()
    label = chunk.label.strip()
    if label and not refs.is_service_marker(label) and label not in title:
        return f"{title}, {label}"
    return title


def _source(chunk: chunking.Chunk, meta: chunking.DocMeta, name: str, period: dict) -> dict:
    """Представляет чанк и метаданные команды в формате для генерации."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.doc_id,
        "document_title": meta.citation_title,
        "section": chunk.label,
        "text": chunk.text,
        "body": chunk.body,
        "citation": _citation(meta, chunk),
        "document_type": meta.kind,
        "document_number": meta.number or None,
        "document_date": iso_date(meta.date),
        "edition": meta.edition or None,
        # В DocMeta значение «действует» используется и как значение по умолчанию.
        "status": None if meta.status == "действует" else meta.status,
        "valid_as_of": meta.valid_as_of or None,
        "parent_contract": meta.parent_contract or None,
        "source_path": name,
        "has_injection": chunk.has_injection,
        **period,
    }


def _version(documents: list, chunks: list, metas: dict) -> str:
    """Вычисляет версию по содержимому, метаданным и нарезке корпуса."""
    payload = {
        "schema": "grounded-context-v2",
        "documents": [(doc.name, doc.text) for doc in documents],
        "chunks": [chunk.to_dict() for chunk in chunks],
        "metadata": {key: asdict(value) for key, value in metas.items()},
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass
class Corpus:
    """Хранит предоставленные командой чанки и проверенный реестр источников."""

    chunks: list
    metas: dict
    sources: dict[str, dict]
    version: str

    @classmethod
    def from_directory(cls, data_dir: Path) -> "Corpus":
        """Читает ZIP через модули команды и строит реестр фрагментов."""
        archive = ingest.find_knowledge_base(data_dir)
        documents = ingest.load_documents_from_zip(archive)
        chunks, metas = chunking.build_chunks(documents, verbose=False)
        names = {doc.doc_id: doc.name for doc in documents}
        periods = {doc.doc_id: document_period(doc.text, metas[doc.doc_id]) for doc in documents}
        sources = {
            chunk.chunk_id: _source(chunk, metas[chunk.doc_id], names[chunk.doc_id], periods[chunk.doc_id])
            for chunk in chunks
        }
        if not chunks or len(sources) != len(chunks):
            raise ValueError("Корпус пуст или содержит повторяющиеся chunk_id")
        return cls(chunks, metas, sources, _version(documents, chunks, metas))

    @property
    def catalog(self) -> list[dict]:
        """Перечисляет реальные документы, позволяя модели заметить пробелы базы."""
        keys = ("document_id", "document_title", "document_number", "edition",
                "effective_from", "effective_to", "valid_as_of", "parent_contract")
        documents = {}
        for source in self.sources.values():
            documents.setdefault(source["document_id"], {key: source.get(key) for key in keys})
        return list(documents.values())

    @cached_property
    def document_index(self):
        """Индексирует документы целиком для поиска регламента по смыслу вопроса."""
        grouped = {}
        for chunk in self.chunks:
            grouped.setdefault(chunk.doc_id, []).append(chunk)
        documents = [replace(items[0], text=items[0].prefix + "\n" +
                             "\n".join(item.body for item in items)) for items in grouped.values()]
        return search.build_index(documents)

    @cached_property
    def content_index(self):
        """Ищет по содержанию пунктов, не повторяя реквизиты в каждом чанке."""
        return search.build_index([replace(chunk, text=chunk.section + "\n" + chunk.body)
                                   for chunk in self.chunks])

    def write_chunks(self, path: Path) -> None:
        """Сохраняет исходные чанки в JSONL для инструментов команды."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            for chunk in self.chunks:
                target.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
