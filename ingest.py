from __future__ import annotations

import hashlib
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import config

TEXT_SUFFIXES = {".txt", ".md", ".csv"}


@dataclass
class RawDoc:

    doc_id: str      # стабильный id: попадает в индекс, логи и ссылки на источники
    name: str        # имя файла внутри архива (для отладки)
    title: str       # читаемое название без служебного префикса группы
    group: str       # "документы компании" | "законы" | "прочее"
    text: str        # нормализованный текст документа
    text_hash: str   # короткий хеш текста: видно, изменился ли документ


# ---------------------------------------------------------------------------
# 1. Где лежит архив
# ---------------------------------------------------------------------------
def find_knowledge_base(data_dir: Path) -> Path:
    for name in config.KB_ZIP_NAMES:
        candidate = data_dir / name
        if candidate.is_file():
            return candidate

    zips = sorted(data_dir.glob("*.zip")) if data_dir.is_dir() else []
    if len(zips) == 1:
        return zips[0]

    raise FileNotFoundError(
        f"Не найден архив базы знаний в {data_dir.resolve()} "
        f"(ожидались: {', '.join(config.KB_ZIP_NAMES)})"
    )


# ---------------------------------------------------------------------------
# 2. Нормализация текста
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Привести текст к предсказуемому виду.

    Четыре действия — и каждое закрывает конкретную будущую проблему:

    * BOM (\\ufeff) — невидимый символ в начале файла. Если его не убрать,
      первая строка не совпадёт с регуляркой, и мы молча не увидим заголовок.
    * \\r\\n -> \\n — в выгрузках встречаются оба варианта переводов строк.
      Иначе один и тот же абзац будет считаться разными строками.
    * неразрывный пробел -> обычный: иначе поиск по «450 000» не найдёт
      «450 000», а это ровно те числа, которые проверяет судья.
    * 3+ пустых строки -> 1: в законах много «воздуха», он раздувает фрагменты
      и мешает определять границы абзацев.
    """
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u2009", " ").replace("\u202f", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)      # хвостовые пробелы в строках
    text = re.sub(r"\n{3,}", "\n\n", text)      # лишние пустые строки
    return text.strip()


def decode_bytes(raw: bytes, name: str) -> str:
    """Декодировать байты файла в текст.

    Сначала utf-8-sig: он «съедает» BOM, если он есть, и работает как обычный
    utf-8, если его нет. Это безопасный вариант по умолчанию.
    Второй шанс — cp866: так иногда «выглядят» имена и содержимое архивов,
    собранных на Windows. Третий — просто utf-8 с заменой битых символов,
    чтобы один битый файл не обрушил весь прогон.
    """
    for encoding in ("utf-8-sig", "cp866"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    print(f"[ingest] файл прочитан с потерями (битая кодировка): {name}", file=sys.stderr)
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 3. Разбор имени файла
# ---------------------------------------------------------------------------
def split_name(name: str) -> tuple[str, str, str]:
    path = Path(name)
    suffix = path.suffix.lower()
    stem = path.stem

    group, sep, rest = stem.partition("__")
    if not sep:
        group, rest = "прочее", stem

    title = rest.replace("_", " ").strip()
    title = re.sub(r"\s{2,}", " ", title)
    return group.strip().lower(), title, suffix


TRANSLIT = str.maketrans(
    {
        "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
        "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
        "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
        "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
        "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    }
)


def doc_id_from_name(name: str) -> str:
    _, title, _ = split_name(name)
    slug = title.lower().translate(TRANSLIT)
    slug = re.sub(r"[^0-9a-z]+", "-", slug).strip("-")[:48].strip("-")
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:6]
    return f"{slug or 'doc'}-{digest}"


# ---------------------------------------------------------------------------
# 4. Чтение архива
# ---------------------------------------------------------------------------
def load_documents_from_zip(zip_path: Path) -> list[RawDoc]:
    documents: list[RawDoc] = []

    with zipfile.ZipFile(zip_path) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or "__MACOSX" in name:
                continue

            group, title, suffix = split_name(name)
            if suffix not in TEXT_SUFFIXES:
                print(f"[ingest] пропущен (не текст): {name}", file=sys.stderr)
                continue

            raw = archive.read(name)
            text = normalize_text(decode_bytes(raw, name))
            if not text:
                print(f"[ingest] пропущен (пустой): {name}", file=sys.stderr)
                continue

            documents.append(
                RawDoc(
                    doc_id=doc_id_from_name(name),
                    name=name,
                    title=title,
                    group=group,
                    text=text,
                    text_hash=hashlib.sha1(text.encode("utf-8")).hexdigest()[:12],
                )
            )

    if not documents:
        raise ValueError(f"В архиве {zip_path} не найдено ни одного читаемого документа")
    return documents


def summarize(documents: list[RawDoc]) -> str:
    """Короткая сводка: сколько документов и символов, с разбивкой по группам."""
    total = sum(len(d.text) for d in documents)
    lines = [f"Документов: {len(documents)}, символов: {total:,}".replace(",", " ")]
    for group in sorted({d.group for d in documents}):
        subset = [d for d in documents if d.group == group]
        chars = sum(len(d.text) for d in subset)
        lines.append(f"  {group}: {len(subset)} док., {chars:,} симв.".replace(",", " "))
    return "\n".join(lines)
