"""Проверки общего движка для веба, CSV и резервного пути."""
import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx

from API.batch import run
from API.engine import LegalAssistant
from platform_solution import solution as engine
from test_web_app import FakeProxy

ROOT = Path(__file__).resolve().parents[1]


class UnifiedPipelineTests(unittest.TestCase):
    """Проверяет общий ответ и источники, не обращаясь к внешним сервисам."""

    def test_batch_and_web_use_same_primary_answer(self):
        """Одному вопросу с одинаковым прокси соответствуют одинаковые ответы."""
        proxy = FakeProxy(lambda _: "ОТВЕТ: Нужно согласование (п. 3.1 Регламента).")
        settings = engine.Settings(data_dir=ROOT / "data", base_url="http://proxy", api_key="test",
                                   use_embeddings=False, use_reranker=False, workers=1)
        assistant = LegalAssistant(settings, httpx.MockTransport(proxy), background_index=False)
        question = "Нужно ли согласование финансового директора для договора на 12 млн рублей?"
        web = assistant.ask(question, [])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "questions.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "question"])
                writer.writeheader()
                writer.writerow({"id": "1", "question": question})
            batch_settings = replace(settings, output=root / "submission.csv", trace=root / "trace.jsonl")
            with patch("API.batch.LegalAssistant", return_value=assistant):
                outcomes = run(batch_settings, path)
            self.assertEqual(web["answer"], outcomes[0].answer)
            self.assertEqual(web["mode"], outcomes[0].mode)
            self.assertEqual(next(csv.DictReader((root / "submission.csv").open()))["answer"], web["answer"])

    def test_recovery_requires_all_cited_sources_to_match(self):
        """Резерв допускается только при совпадении всех подтверждённых источников."""
        settings = engine.Settings(data_dir=ROOT / "data", base_url="http://proxy", api_key="test",
                                   use_embeddings=False)
        assistant = LegalAssistant(settings, httpx.MockTransport(FakeProxy(lambda _: "")), background_index=False)
        self.addCleanup(assistant.close)
        question = engine.Question.parse("q", "Вопрос")
        failure = engine.Outcome("q", "Не удалось", "extractive-llm-error")
        source = {"chunk_id": "source", "source_path": next(name for name, _ in engine.read_sources(ROOT / "data" / "knowledge-base.zip")
                                      if engine._split_name(name) == (assistant.kb.documents[0].group, assistant.kb.documents[0].name)),
                  "section": "п. 1", "text": "Подтверждённое положение."}
        result = {"answer": "Проверенный ответ.", "sources": [source],
                  "claims": [{"chunk_ids": ["source"]}]}
        with patch.object(assistant.assistant, "answer", return_value=failure), \
             patch("API.engine.recover", return_value=result):
            outcome = assistant.answer(question)
            self.assertEqual(outcome.mode, "modular-recovery")
            self.assertIn("Подтверждённое положение", assistant._sources(outcome)[0]["excerpt"])
            result["claims"][0]["chunk_ids"].append("missing")
            self.assertIs(assistant.answer(question), failure)


if __name__ == "__main__":
    unittest.main()
