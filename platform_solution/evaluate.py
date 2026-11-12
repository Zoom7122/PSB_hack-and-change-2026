"""Локальная оценка ответов в духе LLM-судьи платформы (нужны LLM_BASE_URL/LLM_API_KEY/LLM_MODEL).

Сравнивает submission.csv с эталоном (по умолчанию tests/demo_output.csv), как судья:
оценка 0..1 относительно эталона, выдуманный факт обнуляет вопрос.

    python evaluate.py --submission submission.csv --golden tests/demo_output.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import solution as s

JUDGE_PROMPT = """Ты строгий судья юридического ассистента. Сравни ответ команды с эталонным ответом на вопрос.
Оцени по шкале: 1.0 — по сути совпадает с эталоном (верные факты, вывод, ссылки); 0.75 — верно, но неполно;
0.5 — частично верно; 0.25 — в основном неверно или уклончиво; 0.0 — неверно.
Если в ответе есть факт (число, срок, сумма, пункт, вывод), противоречащий эталону или отсутствующий в нём
и явно выдуманный, поставь fabricated=true (это обнуляет оценку).
Верни только JSON: {"score": 0.0, "fabricated": false, "comment": "кратко"}"""


def read(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {r["id"].strip(): r["answer"] for r in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default="submission.csv")
    parser.add_argument("--golden", default=str(Path(__file__).parent / "tests" / "demo_output.csv"))
    parser.add_argument("--questions", default=str(Path(__file__).parent / "tests" / "demo_questions.csv"))
    args = parser.parse_args()
    settings = s.Settings.from_env()
    api = s.Api(settings)
    answers, golden = read(Path(args.submission)), read(Path(args.golden))
    questions = {q.qid: q.text for q in s.load_questions(Path(args.questions))}
    total, rows = 0.0, []
    for qid, gold in golden.items():
        ours = answers.get(qid, "")
        content = (f"Вопрос:\n{questions.get(qid, '')}\n\nЭталон:\n{gold}\n\nОтвет команды:\n{ours}")
        raw = api.chat([{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": content}])
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        verdict = json.loads(match.group()) if match else {"score": 0.0, "fabricated": False, "comment": raw[:200]}
        score = 0.0 if verdict.get("fabricated") else float(verdict.get("score", 0.0))
        total += score
        rows.append((qid, score, verdict.get("comment", "")))
        print(f"{qid:>4}  {score:.2f}  {verdict.get('comment', '')[:150]}")
    print(f"\nСредний балл: {total / max(1, len(rows)):.3f} по {len(rows)} вопросам")
    api.close()


if __name__ == "__main__":
    main()
