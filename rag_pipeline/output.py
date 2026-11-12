"""Запись платформенного CSV и журнала происхождения ответов."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path


def _write_atomic(path: Path, writer) -> None:
    """Заменяет файл только после успешной записи полного содержимого."""
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         dir=path.parent, delete=False) as target:
            name = target.name
            writer(target)
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def write_submission(records: list[dict], path: Path) -> None:
    """Записывает ответы в платформенном формате и исходном порядке."""
    def write(target) -> None:
        """Записывает заголовок и все строки ответов."""
        writer = csv.writer(target, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(["id", "answer"])
        writer.writerows((record["id"], record["answer"]) for record in records)

    _write_atomic(path, write)


def write_trace(records: list[dict], path: Path) -> None:
    """Записывает журнал проверок в JSONL без ключей провайдера."""
    def write(target) -> None:
        """Сохраняет по одному JSON-объекту для каждого вопроса."""
        for record in records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_atomic(path, write)
