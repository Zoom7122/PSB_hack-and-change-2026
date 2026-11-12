"""Векторный индекс: эмбеддинги фрагментов и поиск по косинусному сходству.

Зачем векторы, если уже есть BM25
=================================
BM25 сравнивает слова буквально. Он не найдёт нужный пункт, если вопрос
сформулирован другими словами:

    вопрос:  «грузовик приезжает в два часа ночи — нас должны пропускать?»
    документ: «Арендодатель обязан обеспечить беспрепятственный доступ
               Арендатора и его транспорта к Помещению круглосуточно»

Общих слов почти нет, но смысл совпадает. Эмбеддинг — это вектор смысла:
у этих двух текстов он будет близким. На демо-наборе такой вопрос у BM25
стоял на 14-м месте.

Обратная сторона: векторы плохо различают точные номера и мелкие различия
(«П-17/2024» и «ПТ-05/2023», «10 000 000» и «15 000 000»). Поэтому мы НЕ
заменяем один поиск другим, а объединяем (см. hybrid.py).

Важное ограничение: индекс нельзя переносить между моделями
==========================================================
Локально эмбеддинги считает одна модель (например, OpenAI small, 1536 чисел),
на платформе — другая (bge-m3, 1024 числа). Векторы разных моделей несравнимы,
их размерности не совпадают.

Отсюда правило: **индекс — это кэш, а не часть решения**. В ZIP-сабмите он
опционален и всегда проверяется на совпадение модели. При первом запуске на
платформе решение просто построит индекс заново (~20–40 секунд из 20 минут).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__:
    from . import config
    from .chunking import Chunk
else:
    import config
    from chunking import Chunk


@dataclass
class VectorHit:
    """Результат поиска по векторам."""

    position: int
    chunk: Chunk
    score: float


class VectorIndex:
    """Матрица векторов фрагментов и поиск по ней.

    Устройство простое: матрица (число фрагментов × размерность), каждая строка
    — нормированный вектор фрагмента. Поиск = умножить вектор вопроса на матрицу
    и взять top-k. Для 360 фрагментов это микросекунды, никакой ANN не нужен.
    """

    def __init__(self, chunks: list[Chunk], matrix: np.ndarray) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"Ожидалась матрица (N, d), получено {matrix.shape}")
        if matrix.shape[0] != len(chunks):
            raise ValueError(
                f"Векторов {matrix.shape[0]}, а фрагментов {len(chunks)} — индекс не соответствует базе"
            )
        self.chunks = chunks
        self.matrix = matrix.astype(np.float32)

    # ---------------------------------------------------------------- свойства
    @property
    def dim(self) -> int:
        return int(self.matrix.shape[1])

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------- поиск
    def search_vector(self, query_vector: list[float] | np.ndarray, k: int = 5) -> list[VectorHit]:
        """Найти top-k фрагментов по вектору запроса.

        Косинус равен скалярному произведению, потому что все векторы
        нормированы заранее. Считаем сразу для всех строк матрицы.
        """
        vector = np.asarray(query_vector, dtype=np.float32).ravel()
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Размерность запроса {vector.shape[0]} ≠ размерности индекса {self.dim}. "
                "Скорее всего, индекс собран другой моделью эмбеддингов."
            )
        norm = float(np.linalg.norm(vector)) or 1.0
        scores = self.matrix @ (vector / norm)
        # argsort по возрастанию, берём хвост и переворачиваем — так быстрее,
        # чем сортировать весь массив по убыванию
        top = np.argsort(scores)[-k:][::-1]
        return [
            VectorHit(position=int(i), chunk=self.chunks[int(i)], score=float(scores[int(i)]))
            for i in top
        ]

    # ------------------------------------------------------- сохранение/загрузка
    def save(self, directory: Path | str = config.ARTIFACTS_DIR, *, model: str,
             text_hashes: dict[str, str] | None = None) -> Path:
        """Сохранить индекс в .npz + метаданные в .json.

        В метаданные пишем модель и размерность: без них невозможно понять,
        подходит ли индекс к текущему окружению.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "vector_index.npz"

        chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        np.savez_compressed(path, matrix=self.matrix, chunk_ids=np.array(chunk_ids, dtype=object))

        meta = {
            "model": model,
            "dim": self.dim,
            "count": len(self.chunks),
            "chunk_ids_sha1": hashlib.sha1("\n".join(chunk_ids).encode("utf-8")).hexdigest()[:12],
            "source_hashes": text_hashes or {},
        }
        (directory / "vector_index.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @staticmethod
    def load(chunks: list[Chunk], directory: Path | str = config.ARTIFACTS_DIR,
             *, model: str) -> "VectorIndex":
        """Загрузить индекс, если он существует и подходит к текущей базе и модели.

        Пять обязательных проверок. Любая не прошла — индекс считаем негодным
        и (см. build_or_load) собираем заново. Это дешевле, чем молча работать
        на неправильных векторах и получать плохие ответы без понятной причины.
        """
        directory = Path(directory)
        npz_path = directory / "vector_index.npz"
        meta_path = directory / "vector_index.json"
        if not npz_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError("Индекс не найден")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if meta.get("model") != model:
            raise ValueError(f"Индекс собран моделью {meta.get('model')!r}, а сейчас {model!r}")
        if meta.get("count") != len(chunks):
            raise ValueError(f"В индексе {meta.get('count')} фрагментов, в базе {len(chunks)}")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        current_sha1 = hashlib.sha1("\n".join(chunk_ids).encode("utf-8")).hexdigest()[:12]
        if meta.get("chunk_ids_sha1") != current_sha1:
            raise ValueError("Набор фрагментов изменился — индекс устарел")

        with np.load(npz_path, allow_pickle=True) as data:
            matrix = data["matrix"]
        if matrix.shape[1] != meta.get("dim"):
            raise ValueError("Размерность матрицы не совпадает с метаданными")

        return VectorIndex(chunks, matrix)


def build_vector_index(chunks: list[Chunk], client, *, batch_size: int | None = None,
                       verbose: bool = True) -> VectorIndex:
    """Посчитать эмбеддинги всех фрагментов и собрать индекс.

    Векторизуем chunk.text — то есть шапку вместе с телом (как и в BM25):
    шапка содержит номер документа и реквизиты, которые тоже несут смысл.
    """
    if not chunks:
        raise ValueError("Нечего индексировать: список фрагментов пуст")

    texts = [chunk.text for chunk in chunks]
    if verbose:
        print(f"Считаю эмбеддинги: {len(texts)} фрагментов, модель {config.EMB_MODEL}")

    # Документы кодируем БЕЗ префикса-инструкции — так требуют модели Qwen3
    # и так безопасно для остальных
    embed_documents = getattr(client, "embed_documents", None)
    vectors = (embed_documents(texts, batch_size=batch_size) if embed_documents
               else client.embed(texts, batch_size=batch_size))
    matrix = np.asarray(vectors, dtype=np.float32)

    # Нормировка ещё раз: даже если провайдер вернул ненормированные векторы,
    # дальше всё будет работать в режиме «косинус = скалярное произведение».
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = matrix / norms

    if verbose:
        print(f"Готово: матрица {matrix.shape[0]} × {matrix.shape[1]}")
    return VectorIndex(chunks, matrix)


def build_or_load(chunks: list[Chunk], client, *, model: str,
                  directory: Path | str = config.ARTIFACTS_DIR) -> tuple[VectorIndex, bool]:
    """Загрузить индекс из кэша или построить заново.

    Возвращает (индекс, был_ли_пересчитан). Причина сборки печатается, чтобы
    при разборе логов было видно, почему пришлось считать 20 тысяч векторов.
    """
    try:
        index = VectorIndex.load(chunks, directory, model=model)
        print(f"Индекс загружен из кэша: {len(index)} фрагментов, dim={index.dim}, модель {model}")
        return index, False
    except (FileNotFoundError, ValueError) as exc:
        print(f"Кэш индекса не использован: {exc}")
        index = build_vector_index(chunks, client)
        text_hashes = {chunk.doc_id: chunk.chunk_id for chunk in chunks}
        index.save(directory, model=model, text_hashes=text_hashes)
        return index, True
