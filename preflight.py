"""Финальная проверка всего решения перед отправкой.

Запуск:
    python preflight.py              # полная проверка (около 1–2 минут)
    python preflight.py --fast       # без симуляции платформы

Что проверяется
===============
1. Окружение: версия Python, установленные зависимости против requirements.txt.
2. Все офлайновые проверки шагов 1–7 и самотест подбора весов.
3. Синхронность файлов (check_sync).
4. Конвейер целиком в режиме --dry-run: читает вопросы, ищет, отвечает,
   пишет submission.csv, проверяет формат.
5. Сборка ZIP и его состав по требованиям регламента.
6. Симуляция платформы: архив распаковывается в чистый каталог, данные кладутся
   в DATA_DIR, прогон запускается с недоступным прокси. Цель — убедиться, что
   даже при полном отказе моделей submission.csv создаётся и валиден
   (именно этот сценарий раньше приводил к провалу сабмита).

Каждая проверка запускается отдельным процессом: падение одной не мешает
остальным, а код возврата виден отдельно.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parent
PY = sys.executable


# Вывод в консоль: если поток перенаправлен (pipe/файл), Windows использует
# кодировку локали (cp1251/cp866), и печать символов «→», «×» роняет скрипт
# с UnicodeEncodeError. Переключаем такой поток на UTF-8; интерактивную
# консоль не трогаем, чтобы в ней осталась родная кодировка.
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and not _stream.isatty():
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0
    skipped: bool = False


def run(cmd: list[str], cwd: Path = BASE, env: dict | None = None,
        timeout: int = 600) -> tuple[int, str]:
    """Запустить команду, вернуть (код, объединённый вывод)."""
    merged = os.environ.copy()
    # Принудительно UTF-8 для дочерних процессов: иначе вывод с «→» и «×»
    # падает на Windows с UnicodeEncodeError при перенаправлении в pipe.
    merged["PYTHONIOENCODING"] = "utf-8"
    merged["PYTHONUTF8"] = "1"
    if env:
        merged.update(env)
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), env=merged, timeout=timeout,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, f"таймаут {timeout} с"


def tail(text: str, limit: int = 600) -> str:
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


# --------------------------------------------------------------------------- #
# ПРОВЕРКИ
# --------------------------------------------------------------------------- #

def check_environment() -> Result:
    """Проверить, что нужные библиотеки стоят и что версии не ниже запиненных.

    Расхождение версий — предупреждение, а не провал: на платформе зависимости
    ставятся ровно из requirements.txt, а локально версия может отличаться
    (например, numpy новее). Провал только если библиотеки нет вовсе или
    локальная версия СТАРШЕ запиненной — тогда поведение может отличаться.
    """
    started = time.time()
    major, minor = sys.version_info[:2]
    problems: list[str] = []
    warnings: list[str] = []

    if (major, minor) < (3, 10):
        problems.append(f"Python {major}.{minor} — нужен 3.10+")

    needed = {"numpy": "numpy", "httpx": "httpx", "rank_bm25": "rank-bm25"}
    for module, package in needed.items():
        try:
            __import__(module)
        except ImportError:
            problems.append(f"не установлен {package}")

    def installed_version(package: str) -> str:
        """Версия через метаданные пакета: у rank_bm25 нет атрибута __version__."""
        try:
            from importlib.metadata import version
            return version(package)
        except Exception:      # noqa: BLE001 — метаданных может не быть
            return "?"

    def as_tuple(value: str) -> tuple:
        return tuple(int(part) for part in re.findall(r"\d+", value)[:3])

    requirements = BASE / "requirements.txt"
    if not requirements.exists():
        problems.append("нет requirements.txt")
    else:
        for line in requirements.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "==" not in line:
                continue
            name, pinned = line.split("==", 1)
            package, pinned = name.strip(), pinned.strip()
            module = {"rank-bm25": "rank_bm25"}.get(package.lower(), package.lower())
            try:
                __import__(module)
            except ImportError:
                continue        # уже посчитано выше
            found = installed_version(package)
            if found == "?" or found == pinned:
                continue
            if as_tuple(found) < as_tuple(pinned):
                problems.append(f"{package}: установлено {found}, требуется {pinned} или новее")
            else:
                warnings.append(f"{package} {found} (в requirements {pinned})")

    detail = f"Python {major}.{minor}.{sys.version_info[2]}"
    if warnings:
        detail += " | локально: " + "; ".join(warnings)
    if problems:
        detail += " | ОШИБКИ: " + "; ".join(problems)
    return Result("окружение и зависимости", not problems, detail, time.time() - started)


def check_script(name: str, script: str, marker: str = "ИТОГ",
                 optional: bool = False) -> Result:
    started = time.time()
    if not (BASE / script).exists():
        if optional:
            return Result(name, True,
                          f"нет файла {script} — проверка пропущена "
                          f"(скопируйте его из инструкции, если нужна)", time.time() - started,
                          skipped=True)
        return Result(name, False, f"нет файла {script}", time.time() - started)
    code, output = run([PY, script])
    line = next((ln.strip() for ln in output.splitlines() if ln.strip().startswith(marker)),
                tail(output, 200))
    return Result(name, code == 0, line, time.time() - started)


def check_dry_run(tmp: Path) -> Result:
    started = time.time()
    out = tmp / "preflight_submission.csv"
    code, output = run([PY, "solution.py", "--dry-run", "--out", str(out), "--workers", "2"])
    if code != 0:
        return Result("конвейер (dry-run)", False, tail(output), time.time() - started)
    if not out.exists():
        return Result("конвейер (dry-run)", False, "submission.csv не создан", time.time() - started)
    code2, output2 = run([PY, "-c", (
        "import csv,sys;"
        f"rows=list(csv.DictReader(open(r'{out}',encoding='utf-8')));"
        "assert rows and list(rows[0].keys())==['id','answer'], rows[0].keys();"
        "assert all(r['answer'].strip() for r in rows);"
        "assert max(len(r['answer'].split()) for r in rows)<=150;"
        "print(f'строк {len(rows)}, колонки id,answer, максимум слов {max(len(r[chr(97)+chr(110)+chr(115)+chr(119)+chr(101)+chr(114)].split()) for r in rows)}')"
    )])
    return Result("конвейер (dry-run)", code2 == 0, tail(output2, 300) or tail(output, 300),
                  time.time() - started)


def check_zip() -> Result:
    started = time.time()
    code, output = run([PY, "make_zip.py"])
    if code != 0:
        return Result("сборка ZIP", False, tail(output), time.time() - started)
    archive = BASE / "submission.zip"
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        nested = [n for n in names if "/" in n]
        size_mb = archive.stat().st_size / 1024 / 1024
        detail = f"{len(names)} файлов, {size_mb:.2f} МБ, вложенных папок: {len(nested)}"
        ok = not nested and "solution.py" in names and "requirements.txt" in names
        ok = ok and size_mb < 128
    return Result("сборка ZIP", ok, detail, time.time() - started)


def check_platform_simulation(timeout: int = 180) -> Result:
    """Распаковать архив в чистый каталог и запустить как на платформе.

    Прокси намеренно недоступен: проверяем, что при полном отказе моделей
    решение всё равно создаёт валидный submission.csv (а не падает).
    """
    started = time.time()
    sandbox = Path(tempfile.mkdtemp(prefix="psb_platform_"))
    try:
        with zipfile.ZipFile(BASE / "submission.zip") as zf:
            zf.extractall(sandbox)

        data_dir = sandbox / "data"
        data_dir.mkdir()
        shutil.copy(BASE / "data" / "knowledge-base.zip", data_dir / "knowledge-base.zip")
        shutil.copy(BASE / "data" / "demo_questions.csv", data_dir / "questions.csv")

        env = {
            "LLM_BASE_URL": "http://127.0.0.1:9",
            "EMB_BASE_URL": "http://127.0.0.1:9",
            "LLM_API_KEY": "preflight",
            "EMB_API_KEY": "preflight",
            "LLM_MODEL": "deepseek/deepseek-v4.1-flash",
            "EMB_MODEL": "baai/bge-m3",
            "MAX_WORKERS": "2",
            "TOP_K": "30",
            "DATA_DIR": str(data_dir),
        }
        code, output = run([PY, "solution.py", "--budget", "45", "--workers", "2"],
                           cwd=sandbox, env=env, timeout=timeout)
        submission = sandbox / "submission.csv"
        if not submission.exists():
            return Result("симуляция платформы (прокси недоступен)", False,
                          "submission.csv НЕ создан: " + tail(output, 400), time.time() - started)

        with submission.open(encoding="utf-8") as handle:
            lines = handle.read().strip().splitlines()
        header_ok = lines and lines[0].replace('"', "") == "id,answer"
        rows = len(lines) - 1
        fallbacks = output.count("резервных")
        detail = (f"файл создан: {rows} строк, заголовок {'ок' if header_ok else 'НЕ ок'}; "
                  f"модели недоступны — сработала деградация")
        return Result("симуляция платформы (прокси недоступен)", header_ok and rows > 0,
                      detail, time.time() - started)
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)


def check_no_secrets() -> Result:
    """В архиве не должно быть ключей, паролей и эталонных ответов."""
    started = time.time()
    patterns = {
        "ключ вида sk-…": r"sk-[A-Za-z0-9]{16,}",
        "присвоение ключа": r"(?:api[_-]?key|token|password)\s*=\s*['\"][^'\"]{12,}",
        "эталонные ответы": r"\bGOLDEN\b\s*=\s*\{",
    }
    problems = []
    with zipfile.ZipFile(BASE / "submission.zip") as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            source = zf.read(name).decode("utf-8", errors="ignore")
            for label, pattern in patterns.items():
                if re.search(pattern, source):
                    problems.append(f"{name}: {label}")
    detail = "секретов нет" if not problems else "; ".join(problems)
    return Result("нет ключей и эталонов в архиве", not problems, detail, time.time() - started)


# --------------------------------------------------------------------------- #
# ОТЧЁТ
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="без симуляции платформы")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="psb_preflight_"))
    results: list[Result] = []
    try:
        print("ФИНАЛЬНАЯ ПРОВЕРКА\n")
        steps = [
            ("окружение и зависимости", check_environment),
            ("шаг 1: чтение базы", lambda: check_script("шаг 1: чтение базы", "check_step1.py")),
            ("шаг 2: нарезка", lambda: check_script("шаг 2: нарезка", "check_step2.py")),
            ("шаг 3: BM25", lambda: check_script("шаг 3: BM25", "check_step3.py")),
            ("шаг 4: гибридный поиск", lambda: check_script("шаг 4: гибридный поиск", "check_step4.py")),
            ("шаг 5: разбор вопроса", lambda: check_script("шаг 5: разбор вопроса", "check_step5.py")),
            ("шаг 6: ответ и отказы", lambda: check_script("шаг 6: ответ и отказы", "check_step6.py")),
            ("шаг 7: конвейер", lambda: check_script("шаг 7: конвейер", "check_step7.py")),
            ("подбор весов (самотест)", lambda: check_script_simple(
                "подбор весов (самотест)", ["tune_fusion.py", "--selftest"])),
            ("синхронность файлов", lambda: check_script(
                "синхронность файлов", "check_sync.py", optional=True)),
            ("конвейер целиком (dry-run)", lambda: check_dry_run(tmp)),
            ("сборка ZIP", check_zip),
            ("нет ключей и эталонов", check_no_secrets),
        ]
        if not args.fast:
            steps.append(("симуляция платформы", check_platform_simulation))

        for index, (name, func) in enumerate(steps, start=1):
            result = func()
            results.append(result)
            mark = "ПРОПУСК" if result.skipped else ("OK  " if result.ok else "ПРОВАЛ")
            print(f"{index:2d}. {mark} {result.name:34s} {result.seconds:5.1f} с  {result.detail[:70]}")

        # ------------------------------------------------------------- вердикт
        failed = [r for r in results if not r.ok]
        skipped = [r for r in results if r.skipped]
        total_time = sum(r.seconds for r in results)
        print("\n" + "=" * 78)
        print(f"Проверок: {len(results)} | пройдено: {len(results) - len(failed) - len(skipped)} | "
              f"пропущено: {len(skipped)} | провалено: {len(failed)} | время: {total_time:.0f} с")

        report = tmp / "preflight_report.txt"
        report.write_text("\n".join(
            f"{'OK' if r.ok else 'FAIL'}\t{r.name}\t{r.detail}" for r in results), encoding="utf-8")

        if failed:
            print("\nЧТО ИСПРАВИТЬ:")
            for r in failed:
                print(f"  - {r.name}: {r.detail}")
            return 1

        archive = BASE / "submission.zip"
        print(f"\nВСЁ ГОТОВО. Файл для загрузки: {archive}")
        print(f"Размер: {archive.stat().st_size / 1024:.1f} КБ")
        print("\nОстаётся: загрузить этот ZIP на платформу и дождаться оценки.")
        print("Сабмитов 5 в день (сброс в 00:00 МСК).")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_script_simple(name: str, args: list[str]) -> Result:
    """Запуск скрипта с аргументами (для самотеста подбора весов)."""
    started = time.time()
    code, output = run([PY] + args)
    line = next((ln.strip() for ln in output.splitlines() if "ИТОГ" in ln), tail(output, 200))
    return Result(name, code == 0, line, time.time() - started)


if __name__ == "__main__":
    sys.exit(main())
