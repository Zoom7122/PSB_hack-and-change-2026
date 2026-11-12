"""Загрузка корпуса и доступ к подлинным фрагментам для проверки ссылок."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)

from data_processing import chunking, ingest, refs  # noqa: E402


def _citation(meta: chunking.DocMeta, chunk: chunking.Chunk) -> str:
    """Создаёт ссылку только из метаданных исходного документа."""
    title = meta.citation_title.strip()
    label = chunk.label.strip()
    if label and not refs.is_service_marker(label) and label not in title:
        return f"{title}, {label}"
    return title


def _source(chunk: chunking.Chunk, meta: chunking.DocMeta, name: str) -> dict:
    """Представляет чанк и метаданные команды в формате для генерации."""
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.doc_id,
        "document_title": meta.citation_title,
        "section": chunk.label,
        "text": chunk.text,
        "citation": _citation(meta, chunk),
        "document_type": meta.kind,
        "document_number": meta.number or None,
        "edition": meta.edition or None,
        # В DocMeta значение «действует» используется и как значение по умолчанию.
        "status": None if meta.status == "действует" else meta.status,
        "valid_as_of": meta.valid_as_of or None,
        "parent_contract": meta.parent_contract or None,
        "source_path": name,
        "has_injection": chunk.has_injection,
    }


def _version(documents: list, chunks: list, metas: dict) -> str:
    """Вычисляет версию по содержимому, метаданным и нарезке корпуса."""
    payload = {
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
        sources = {
            chunk.chunk_id: _source(chunk, metas[chunk.doc_id], names[chunk.doc_id])
            for chunk in chunks
        }
        if not chunks or len(sources) != len(chunks):
            raise ValueError("Корпус пуст или содержит повторяющиеся chunk_id")
        return cls(chunks, metas, sources, _version(documents, chunks, metas))

    def write_chunks(self, path: Path) -> None:
        """Сохраняет исходные чанки в JSONL для инструментов команды."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as target:
            for chunk in self.chunks:
                target.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
