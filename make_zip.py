"""Собрать ZIP-сабмит по требованиям регламента.

Запуск:
    python make_zip.py                # собрать submission.zip
    python make_zip.py --check        # только проверить состав, без записи

Что требует регламент (п. 4)
============================
  * файлы в КОРНЕ архива, без внешней папки;
  * solution.py — обязательная точка входа;
  * requirements.txt — только библиотеки из разрешённого списка, с версиями;
  * дополнительные .py-модули — тоже в корне (переносятся только корневые);
  * базу знаний и вопросы в архив НЕ включать;
  * никаких ключей в файлах;
  * вызовы pip install из solution.py запрещены.

Почему отдельный скрипт, а не «собрать руками»
==============================================
Руками легко утащить в архив лишнее (artifacts/ с индексом на 6 МБ, .env с
ключами, __pycache__) или забыть модуль, который импортируется. Тогда ошибка
всплывёт на платформе через 20 минут прогона. Скрипт собирает ровно нужное и
сам проверяет результат.
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "submission.zip"

# Точка входа и зависимости — обязательны.
REQUIRED = ["solution.py", "requirements.txt"]

# Модули, которые импортирует решение. Список получен обходом импортов, а не на
# глаз: если забыть модуль, на платформе будет ModuleNotFoundError.
MODULES = [
    "answering.py",
    "chunking.py",
    "config.py",
    "hybrid.py",
    "ingest.py",
    "llm_client.py",
    "query_prep.py",
    "question_analysis.py",
    "search.py",
    "textproc.py",
    "vector_index.py",
]

# Что в архив не попадает никогда.
FORBIDDEN_NAMES = {".env", ".gitignore"}
FORBIDDEN_SUFFIXES = {".zip", ".pyc", ".jsonl", ".json", ".csv", ".npy"}


def collect() -> tuple[list[Path], list[str]]:
    """Собрать список файлов и список проблем."""
    problems: list[str] = []
    files: list[Path] = []

    for name in REQUIRED + MODULES:
        path = BASE / name
        if not path.exists():
            problems.append(f"нет обязательного файла: {name}")
        else:
            files.append(path)

    # проверка импортов: каждый локальный import должен быть в списке
    local_names = {path.stem for path in files if path.suffix == ".py"}
    stdlib_and_libs = {
        "argparse", "csv", "json", "sys", "threading", "time", "re", "math",
        "hashlib", "dataclasses", "datetime", "pathlib", "zipfile", "io",
        "concurrent", "collections", "typing", "itertools", "functools",
        "numpy", "httpx", "rank_bm25", "tempfile", "traceback", "os",
        "__future__", "random", "string", "textwrap", "difflib", "unicodedata",
    }
    import_pattern = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
    for path in [p for p in files if p.suffix == ".py"]:
        source = path.read_text(encoding="utf-8", errors="ignore")
        for match in import_pattern.finditer(source):
            top = match.group(1).split(".")[0]
            if top in stdlib_and_libs or top in local_names:
                continue
            problems.append(f"{path.name}: импорт «{top}» не входит в архив и не является библиотекой")

    # ключи и пароли в файлах
    secret_pattern = re.compile(r"(sk-[A-Za-z0-9]{16,}|api[_-]?key\s*=\s*['\"][^'\"]{12,})",
                                re.IGNORECASE)
    for path in files:
        source = path.read_text(encoding="utf-8", errors="ignore")
        if secret_pattern.search(source):
            problems.append(f"{path.name}: похоже на настоящий ключ в коде")

    # pip install внутри решения запрещён
    solution = BASE / "solution.py"
    if solution.exists() and re.search(r"pip\s+install|subprocess|os\.system",
                                       solution.read_text(encoding="utf-8", errors="ignore")):
        problems.append("solution.py: вызов установки пакетов или subprocess (регламент запрещает)")

    # зависимости: только из разрешённого списка, с зафиксированными версиями
    allowed_path = BASE / "data" / "requirements_allowed.txt"
    requirements = BASE / "requirements.txt"
    if requirements.exists() and allowed_path.exists():
        allowed = set()
        for line in allowed_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                allowed.add(line.split("==")[0].lower())
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "==" not in line:
                problems.append(f"requirements.txt: «{line}» без зафиксированной версии")
                continue
            name = line.split("==")[0].strip().lower()
            if name not in allowed:
                problems.append(f"requirements.txt: «{name}» нет в разрешённом списке")

    return files, problems


def build(files: list[Path], out: Path) -> None:
    """Записать архив: файлы в корне, без вложенной папки."""
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)      # только имя, без каталогов


def verify_archive(path: Path) -> list[str]:
    """Проверить готовый архив так, как его увидит платформа."""
    problems: list[str] = []
    if not path.exists():
        return [f"архив не создан: {path}"]

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if any("/" in name for name in names):
            problems.append("в архиве есть вложенные папки — файлы должны быть в корне")
        if "solution.py" not in names:
            problems.append("в архиве нет solution.py")
        if "requirements.txt" not in names:
            problems.append("в архиве нет requirements.txt")
        for name in names:
            suffix = Path(name).suffix.lower()
            if name in FORBIDDEN_NAMES or suffix in FORBIDDEN_SUFFIXES:
                problems.append(f"лишний файл в архиве: {name}")
        # код должен компилироваться
        for name in names:
            if name.endswith(".py"):
                try:
                    compile(archive.read(name).decode("utf-8"), name, "exec")
                except SyntaxError as exc:
                    problems.append(f"{name}: синтаксическая ошибка ({exc.msg}, строка {exc.lineno})")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 128:
            problems.append(f"архив {size_mb:.1f} МБ — лимит 128 МБ")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="только проверить, не собирать")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()

    files, problems = collect()
    print(f"Файлов к включению: {len(files)}")
    for path in files:
        print(f"  {path.name:28s} {path.stat().st_size / 1024:7.1f} КБ")

    if problems:
        print("\nПРОБЛЕМЫ ДО СБОРКИ:")
        for problem in problems:
            print(f"  - {problem}")
        if not args.check:
            print("\nАрхив не собран: сначала исправьте проблемы.")
        return 1

    if args.check:
        print("\nПроверка пройдена, архив не собирался (--check).")
        return 0

    out = Path(args.out)
    build(files, out)
    after = verify_archive(out)
    if after:
        print("\nПРОБЛЕМЫ В АРХИВЕ:")
        for problem in after:
            print(f"  - {problem}")
        return 1

    print(f"\nСобран: {out} ({out.stat().st_size / 1024:.1f} КБ)")
    print("Состав: файлы в корне, без вложенной папки, без ключей и без базы знаний.")
    print("Загружать этот файл на платформу как ZIP-сабмит.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
