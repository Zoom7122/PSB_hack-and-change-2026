"""Тонкая обвязка существующих индексаторов команды."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import Corpus
from data_processing import config, hybrid, search, vector_index
from data_processing.llm_client import LLMClient

LOGGER = logging.getLogger(__name__)


def data_dir() -> Path:
    """Возвращает общий для платформы и Makefile каталог входных данных."""
    return Path(os.getenv("DATA_DIR", "data").strip() or "data")


def artifacts_dir() -> Path:
    """Возвращает общий каталог индексов и результатов проверки."""
    return Path(os.getenv("ARTIFACTS_DIR", "artifacts").strip() or "artifacts")


def _cached_vectors(corpus: Corpus, directory: Path, model: str):
    """Загружает индекс команды только для той же версии корпуса и модели."""
    marker = directory / "vector_corpus.json"
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata != {"corpus_version": corpus.version, "model": model}:
            return None
        return vector_index.VectorIndex.load(corpus.chunks, directory, model=model)
    except (OSError, ValueError, KeyError, IndexError):
        return None


def build_vectors(corpus: Corpus, directory: Path, client: LLMClient, *, force: bool = False):
    """Вызывает существующий векторный индексатор команды и сохраняет его версию."""
    model = config.EMB_MODEL
    vectors = None if force else _cached_vectors(corpus, directory, model)
    if vectors is not None:
        return vectors
    vectors = vector_index.build_vector_index(corpus.chunks, client)
    directory.mkdir(parents=True, exist_ok=True)
    vectors.save(directory, model=model)
    marker = {"corpus_version": corpus.version, "model": model}
    (directory / "vector_corpus.json").write_text(
        json.dumps(marker, ensure_ascii=False), encoding="utf-8"
    )
    return vectors


@dataclass
class TeamSearch:
    """Держит поисковый объект команды и закрывает его сетевой клиент."""

    searcher: hybrid.HybridSearcher
    client: LLMClient | None

    def close(self) -> None:
        """Освобождает соединения клиента эмбеддингов и реранкера."""
        if self.client is not None:
            self.client.close()


def create_search(corpus: Corpus, directory: Path, *, force: bool = False,
                  require_vectors: bool = False) -> TeamSearch:
    """Создаёт поиск команды, при сбое эмбеддингов оставляя BM25."""
    lexical = search.build_index(corpus.chunks)
    client = None
    try:
        if not config.EMB_BASE_URL or not config.EMB_API_KEY:
            raise ValueError("Не настроены адрес и ключ эмбеддингов")
        client = LLMClient()
        vectors = build_vectors(corpus, directory, client, force=force)
        return TeamSearch(hybrid.HybridSearcher(lexical, vectors, client), client)
    except Exception as exc:
        if client is not None:
            client.close()
        if require_vectors:
            raise RuntimeError(f"Не удалось построить векторный индекс: {type(exc).__name__}") from None
        LOGGER.warning("Векторный индекс недоступен (%s), используется BM25", type(exc).__name__)
        empty = np.zeros((len(corpus.chunks), 1), dtype=np.float32)
        vectors = vector_index.VectorIndex(corpus.chunks, empty)
        return TeamSearch(hybrid.HybridSearcher(lexical, vectors, None), None)


def main() -> None:
    """Запускает подготовку чанков или существующего индексатора команды."""
    parser = argparse.ArgumentParser(description="Подготовка индексов базы документов")
    parser.add_argument("--chunks-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    corpus = Corpus.from_directory(data_dir())
    directory = artifacts_dir()
    corpus.write_chunks(directory / "chunks.jsonl")
    if not args.chunks_only:
        runtime = create_search(corpus, directory, force=args.force, require_vectors=True)
        runtime.close()
    print(f"Документов: {len(corpus.metas)}, фрагментов: {len(corpus.chunks)}")
    print(f"Версия корпуса: {corpus.version}")


if __name__ == "__main__":
    main()
