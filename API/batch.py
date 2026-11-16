"""Пакетный CSV-запуск общего с вебом конвейера."""
from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from API.engine import LegalAssistant, load_env_file
from platform_solution import solution as engine

LOG = logging.getLogger(__name__)


def run(settings: engine.Settings, questions_path: Path, limit: int | None = None) -> list[engine.Outcome]:
    """Обрабатывает CSV, сохраняя ответы по мере завершения запросов."""
    started = time.monotonic()
    questions = engine.load_questions(questions_path)
    if limit:
        questions = questions[:limit]
    if len({q.qid for q in questions}) != len(questions):
        raise ValueError("ID вопросов должны быть уникальными")
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    engine.write_submission(settings.output, questions, {})
    deadline = started + settings.time_budget
    settings = replace(settings, data_dir=settings.data_dir.resolve())
    assistant = LegalAssistant(settings, background_index=False, deadline=deadline)
    assistant.assistant.deadline = deadline
    if assistant.api:
        assistant.api.deadline = deadline
    answers, outcomes = {}, []
    try:
        with ThreadPoolExecutor(max_workers=settings.workers) as pool:
            futures = {pool.submit(assistant.answer, q, len(questions) - i): q for i, q in enumerate(questions)}
            for future in as_completed(futures):
                outcome = future.result()
                outcomes.append(outcome)
                answers[outcome.qid] = outcome.answer
                engine.write_submission(settings.output, questions, answers)
                LOG.info("[%d/%d] id=%s mode=%s", len(outcomes), len(questions), outcome.qid, outcome.mode)
    finally:
        engine.write_submission(settings.output, questions, answers)
        assistant.close()
    if settings.trace:
        settings.trace.parent.mkdir(parents=True, exist_ok=True)
        order = {q.qid: i for i, q in enumerate(questions)}
        with settings.trace.open("w", encoding="utf-8") as trace:
            for outcome in sorted(outcomes, key=lambda item: order[item.qid]):
                trace.write(json.dumps(outcome.__dict__, ensure_ascii=False) + "\n")
    return outcomes


def main(argv: list[str] | None = None) -> None:
    """Читает параметры пакетного запуска и запускает единый конвейер."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit должен быть положительным")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env_file(Path.cwd() / ".env")
    settings = engine.Settings.from_env()
    settings = replace(settings, data_dir=args.data_dir or settings.data_dir,
                       output=args.out or settings.output, trace=args.trace or settings.trace)
    run(settings, args.questions or settings.data_dir / "questions.csv", args.limit)
