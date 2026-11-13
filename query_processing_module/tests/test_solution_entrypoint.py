"""Проверки CSV-точки входа и нормализации без настоящего LLM."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from API import for_solutions
from API.backend_api import prepare_query
from query_processing_module.query_processing.config import LLMSettings


class BackendApiTests(unittest.TestCase):
    def test_prepare_query_trims_and_preserves_question(self):
        """Обрезает краевые пробелы и проверяет запрос по QueryRequest."""
        request = prepare_query(" \tДОГОВОР расторгнут?  \n")

        self.assertEqual(request.question, "ДОГОВОР расторгнут?")
        self.assertEqual(request.history, [])

    def test_prepare_query_rejects_blank_question(self):
        """Отклоняет вопрос, ставший пустым после нормализации."""
        with self.assertRaises(ValidationError):
            prepare_query(" \t \n")


class FakeLLMClient:
    def __init__(self, settings):
        """Сохраняет ключ для теста маскирования логов."""
        self.settings = settings
        self.closed = False

    async def complete(self, messages):
        """Возвращает фиксированный ответ без сетевого вызова."""
        return "raw model answer service-secret"

    async def close(self):
        """Отмечает закрытие клиента."""
        self.closed = True


class FakeQueryService:
    last_request = None
    last_response = None

    def __init__(self, llm_client):
        """Принимает декорированный LLM-клиент для проверки вызова."""
        self.llm_client = llm_client

    async def process_query(self, request):
        """Сохраняет запрос, вызывает подмену и возвращает тестовый статус."""
        FakeQueryService.last_request = request
        FakeQueryService.last_response = await self.llm_client.complete(
            [{"role": "user", "content": request.question}]
        )
        return {"status": "clarification_required", "message": "Уточните вопрос", "search_request": None}


class SolutionEntrypointTests(unittest.TestCase):
    def test_main_logs_response_and_writes_clarification(self):
        """Сохраняет уточнение Query Processing и маскирует ключ в логе."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            with (data_dir / "questions.csv").open(
                "w", encoding="utf-8", newline=""
            ) as questions_file:
                writer = csv.writer(questions_file)
                writer.writerow(["id", "question"])
                writer.writerow(["case-1", "  ДОГОВОР расторгнут?  "])

            settings = LLMSettings(
                "https://platform.example/v1", "service-secret", "platform-model"
            )
            fake_client = FakeLLMClient(settings)
            environment = {
                **os.environ,
                "DATA_DIR": str(data_dir),
                "QUERY_PROCESSING_MODE": "single",
                "MAX_WORKERS": "1",
                "ARTIFACTS_DIR": str(root / "artifacts"),
                "LLM_LOG_RESPONSES": "1",
            }
            FakeQueryService.last_request = None
            FakeQueryService.last_response = None

            with (
                patch.dict(os.environ, environment, clear=True),
                patch("API.for_solutions.Path.cwd", return_value=root),
                patch("API.for_solutions.SERVICE_TYPES", {"single": FakeQueryService}),
                patch("API.for_solutions.OpenAIClient", return_value=fake_client),
                patch.object(LLMSettings, "from_environment", return_value=settings),
                patch("API.for_solutions.Corpus.from_directory") as fake_corpus,
                patch("API.for_solutions.create_search") as fake_search,
                self.assertLogs("solution", level="INFO") as captured,
            ):
                fake_corpus.return_value.version = "test-corpus"
                for_solutions.main()

            with (root / "submission.csv").open(
                encoding="utf-8", newline=""
            ) as submission_file:
                rows = list(csv.DictReader(submission_file))

        self.assertEqual(rows, [{"id": "case-1", "answer": "Уточните вопрос"}])
        self.assertEqual(FakeQueryService.last_request.question, "ДОГОВОР расторгнут?")
        self.assertEqual(FakeQueryService.last_response, "raw model answer service-secret")
        self.assertTrue(fake_client.closed)
        fake_search.return_value.close.assert_called_once()
        log_text = "\n".join(captured.output)
        self.assertIn('"id": "case-1"', log_text)
        self.assertIn("[REDACTED_SECRET]", log_text)
        self.assertNotIn("service-secret", log_text)

    def test_load_questions_rejects_empty_normalized_question(self):
        """Останавливает обработку пустого вопроса до вызова LLM."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            questions_path = data_dir / "questions.csv"
            questions_path.write_text("id,question\n1,   \n", encoding="utf-8")

            with self.assertRaises(ValidationError):
                for_solutions.load_questions(data_dir)

    def test_load_questions_preserves_id_and_requires_columns(self):
        """Сохраняет ID без нормализации и требует id/question в CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            questions_path = data_dir / "questions.csv"
            questions_path.write_text(
                "id,question\n  case-1  ,Вопрос\n", encoding="utf-8"
            )
            rows = for_solutions.load_questions(data_dir)
            self.assertEqual(rows[0].row_id, "  case-1  ")

            questions_path.write_text("id,prompt\n1,Вопрос\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                for_solutions.load_questions(data_dir)


if __name__ == "__main__":
    unittest.main()
