"""Ручной прогон с настоящей моделью.

    python -m answer_generation examples/rent_question.json --debug

Ключи и модель берутся из .env (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import GenerationSettings, LLMSettings
from .llm_client import OpenAIClient
from .service import AnswerGenerationService


async def _main(paths: list[str], debug: bool) -> int:
    client = OpenAIClient(LLMSettings.from_environment())
    service = AnswerGenerationService(client, GenerationSettings.from_environment())
    try:
        for path in paths:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            outcome = await service.run(payload)
            print(f"=== {path}")
            print(json.dumps(outcome.to_dict(include_debug=debug), ensure_ascii=False, indent=2))
    finally:
        await client.close()
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="Модуль 5: GenerationRequest JSON → GenerationResult")
    parser.add_argument("requests", nargs="+", help="файлы с GenerationRequest")
    parser.add_argument("--debug", action="store_true", help="показать трассировку этапов")
    args = parser.parse_args()
    return asyncio.run(_main(args.requests, args.debug))


if __name__ == "__main__":
    sys.exit(main())
