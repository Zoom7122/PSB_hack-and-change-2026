"""Проверка, что файлы проекта — те же версии, что в инструкции.

Запуск:
    python check_sync.py

Зачем это нужно
===============
Мы правим несколько файлов сразу, и легко получить рассинхрон: `tune_fusion.py`
новый, а `hybrid.py` — старый. Тогда падает не с понятной ошибкой, а с
`TypeError: ... unexpected keyword argument 'weights'`, и выглядит это как баг
кода, хотя это просто не скопированный файл.

Скрипт не сверяет хэши (они разные, если вы что-то форматировали под себя), а
проверяет **наличие нужных возможностей**: есть ли в файле параметр, функция,
константа. Так он не будет ругаться на косметику и поймает реальный рассинхрон.

Два уровня важности
===================
**Обязательные** — без них решение не запускается или сдаётся неверно. Их
отсутствие или устаревание означает провал.

**Необязательные** — удобство разработки: проверки, бенчмарки, сборщик архива,
пример настроек. Их отсутствие не влияет на сабмит: `solution.py` работает и без
них. Итоговый вердикт из-за них не становится отрицательным — иначе «нет
check_embedding_models.py» выглядело бы как блокирующая проблема.

Как читать вывод
================
    OK    — возможность на месте
    СТАРО — файл старой версии, скопируйте его целиком из инструкции
    НЕТ   — файла нет
"""

from __future__ import annotations

import sys

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

from pathlib import Path

BASE = Path(__file__).resolve().parent

# Файлы, без которых решение не работает. Импортируются из solution.py.
REQUIRED_FILES = [
    "solution.py",
    "config.py",
    "llm_client.py",
    "vector_index.py",
    "hybrid.py",
    "answering.py",
    "question_analysis.py",
    "chunking.py",
    "ingest.py",
    "search.py",
    "textproc.py",
    "query_prep.py",
    "requirements.txt",
]

# Файлы для разработки: проверки, бенчмарки, сборка архива, пример настроек.
OPTIONAL_FILES = [
    "check_step1.py", "check_step2.py", "check_step3.py", "check_step4.py",
    "check_step5.py", "check_step6.py", "check_step7.py",
    "tune_fusion.py", "check_embedding_models.py", "build_chunks.py",
    "make_zip.py", "preflight.py", ".env.example",
]

# (файл, что ищем в исходнике, название возможности, обязательность)
CHECKS: list[tuple[str, str, str, bool]] = [
    # --- config.py
    ("config.py", "EMB_QUERY_PREFIX", "префикс-инструкция для запроса", True),
    ("config.py", "BM25_WEIGHT", "вес BM25 при слиянии", True),
    ("config.py", "BM25_FUSION_DEPTH", "глубина BM25 при слиянии", True),
    ("config.py", "EMB_BATCH_SIZE", "размер батча эмбеддингов", True),
    ("config.py", "RERANK_MODE", "режим реранжа", True),
    ("config.py", "слияние", "строка слияния в describe()", False),
    ("config.py", "REFUSAL_BM25_MIN", "порог отказа (шаг 6)", False),
    ("config.py", "ANSWER_MAX_WORDS", "лимит слов в ответе (шаг 6)", False),
    ("config.py", "MAX_WORKERS", "число потоков из окружения платформы", False),
    ("config.py", "SOLVER_WORKERS", "число потоков прогона", False),
    # --- llm_client.py
    ("llm_client.py", "def embed_query", "вектор запроса с префиксом", True),
    ("llm_client.py", "def embed_documents", "векторы документов без префикса", True),
    ("llm_client.py", "def rerank_by_llm", "реранж силами chat-модели", True),
    ("llm_client.py", "MAX_RETRIES", "повторы на 429/5xx", True),
    # --- vector_index.py
    ("vector_index.py", "embed_documents", "документы кодируются без префикса", True),
    ("vector_index.py", "sha1", "проверка кэша по составу фрагментов", True),
    # --- hybrid.py
    ("hybrid.py", "weights: list[float] | None", "параметр weights в RRF", True),
    ("hybrid.py", "depths: list[int | None] | None", "параметр depths в RRF", True),
    ("hybrid.py", "BM25_WEIGHT", "вес BM25 передаётся в слияние", True),
    ("hybrid.py", "embed_query", "запрос кодируется с префиксом", True),
    ("hybrid.py", "except Exception", "деградация при любой ошибке", True),
    # --- шаг 5
    ("question_analysis.py", "def load_doc_meta", "загрузка метаданных с восстановлением", True),
    ("question_analysis.py", "asserted", "различие утверждения и упоминания", True),
    ("question_analysis.py", "1900 <= number <= 2100", "годы не считаются суммами", True),
    # --- шаг 6
    ("answering.py", "_cfg(", "значения шага 6 работают без правок config.py", True),
    ("answering.py", "INJECTION_PATTERNS", "защита от инъекций в документах", True),
    ("answering.py", "def verify_citations", "проверка ссылок на фрагменты", True),
    ("answering.py", "def refusal_decision", "решение об отказе по сигналам поиска", True),
    ("answering.py", "def sanitize_answer", "обрезка по словам и чистка markdown", True),
    ("answering.py", "needs_correction", "исправление только утверждений", False),
    # --- шаг 7
    ("solution.py", "def load_questions", "чтение questions.csv", True),
    ("solution.py", "def validate_submission", "проверка файла перед сдачей", True),
    ("solution.py", "FALLBACK_ANSWER", "резервный ответ при нехватке времени", True),
    ("solution.py", "soft_deadline", "бюджет времени с резервом 15%", True),
    ("solution.py", "empty_vector_index", "деградация до BM25 без эмбеддингов", True),
    ("solution.py", "breaker", "предохранитель при недоступной модели", True),
    ("solution.py", "reconfigure(encoding", "печать не падает на Windows", True),
    ("solution.py", "dry-run", "проверка конвейера без сети", False),
    # --- вспомогательные скрипты (не влияют на сабмит)
    ("check_step4.py", "use_rerank=False", "строка «гибрид (RRF)» без реранжа", False),
    ("check_step4.py", "textproc.tokenize_query", "BM25 без двойного стемминга", False),
    ("check_step4.py", "лучшего из двух", "критерий сравнения с лучшим поиском", False),
    ("build_chunks.py", "config.ARTIFACTS_DIR", "запись в artifacts/, а не build/", False),
    ("check_embedding_models.py", "EMB_QUERY_PREFIX", "бенчмарк моделей эмбеддингов", False),
    ("tune_fusion.py", "BM25_FUSION_DEPTH", "подбор весов слияния", False),
]


def main() -> int:
    blocking: list[str] = []
    warnings: list[str] = []

    # ---------------------------------------------------------------- ФАЙЛЫ
    print("ФАЙЛЫ, БЕЗ КОТОРЫХ РЕШЕНИЕ НЕ РАБОТАЕТ\n")
    for name in REQUIRED_FILES:
        if (BASE / name).exists():
            print(f"  OK     {name}")
        else:
            print(f"  НЕТ    {name}   ← блокирующая проблема")
            blocking.append(name)

    print("\nФАЙЛЫ ДЛЯ РАЗРАБОТКИ (на сабмит не влияют)\n")
    for name in OPTIONAL_FILES:
        if (BASE / name).exists():
            print(f"  OK     {name}")
        else:
            print(f"  нет    {name}   (необязательно)")
            warnings.append(f"нет {name}")

    # ----------------------------------------------------------- ВОЗМОЖНОСТИ
    print("\nВОЗМОЖНОСТИ\n")
    current_file = ""
    for name, needle, title, required in CHECKS:
        path = BASE / name
        if name != current_file:
            print(f"  {name}")
            current_file = name
        if not path.exists():
            level = "НЕТ   " if required else "нет   "
            print(f"    {level} {title}" + ("" if required else "   (необязательно)"))
            (blocking if required else warnings).append(f"{name}: файла нет")
            continue

        source = path.read_text(encoding="utf-8", errors="ignore")
        if needle in source:
            print(f"    OK     {title}")
            continue

        if required:
            print(f"    СТАРО  {title}   (нет «{needle}»)   ← блокирующая проблема")
            blocking.append(f"{name}: старая версия ({title})")
        else:
            print(f"    старо  {title}   (необязательно)")
            warnings.append(f"{name}: старая версия ({title})")

    # ---------------------------------------------------------------- ИТОГ
    print()
    if not (BASE / ".env").exists():
        print("  ВНИМАНИЕ: .env не найден — скопируйте .env.example в .env и впишите ключи")
        print("            (на платформе ключи приходят из окружения, файл не нужен)\n")

    if blocking:
        print("ИТОГ: решение не готово к сдаче.")
        for item in blocking:
            print(f"  - {item}")
        print("\n  Что делать: скопировать эти файлы целиком из инструкции")
        print("  (в документации каждый файл — блок кода с полным содержимым).")
        return 1

    print("ИТОГ: обязательные файлы в порядке — решение готово к сдаче.")
    if warnings:
        print(f"\n  Необязательные замечания ({len(warnings)}):")
        for item in warnings[:8]:
            print(f"  - {item}")
        print("  На работу решения и на сабмит это не влияет.")
        if any("build_chunks.py" in item for item in warnings):
            print("\n  Про build_chunks.py: в старой версии было config.BUILD_DIR,")
            print("  в новой — config.ARTIFACTS_DIR. Скрипт нужен только для создания")
            print("  artifacts/docs.json (ускоряет шаг 5 на секунду). Замена одной строки:")
            print("      out = Path(config.ARTIFACTS_DIR)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
