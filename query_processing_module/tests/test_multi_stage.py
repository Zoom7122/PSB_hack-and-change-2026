"""Проверки многошаговой обработки с подставными ответами LLM."""

import asyncio
import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from query_processing.config import ConfigurationError
from query_processing.llm_client import LLMError, LLMTimeoutError
from query_processing.multi_stage import MultiStageQueryProcessingService
from query_processing.validator import InvalidLLMResponse


class MockLLM:
    def __init__(self, *responses):
        """Сохраняет заготовленные ответы и вызовы модели."""
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages):
        """Возвращает следующий ответ или передаёт подготовленную ошибку."""
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response, ensure_ascii=False)


def context(question="Как расторгнуть договор №123?"):
    """Создаёт корректный результат восстановления контекста."""
    return {"standalone_query": question, "clarification_question": None}


def search():
    """Создаёт пример поискового запроса с номером и датой."""
    return {"standalone_query": "Как расторгнуть договор №123?",
            "search_queries": ["расторжение договора №123", "уведомление о расторжении"],
            "filters": {"document_number": "123", "document_type": "договор", "event_date": "2025-04-15"}}


class MultiStageTests(unittest.IsolatedAsyncioTestCase):
    async def test_ready_uses_three_calls_and_passes_context(self):
        """Передаёт историю и контекст между этапами и сохраняет внешний контракт."""
        mock = MockLLM(context(), {"is_legal": True, "message": None}, search())
        history = [{"role": "user", "content": "Договор №123 от 15.04.2025"}]
        result = await MultiStageQueryProcessingService(mock).process_query({"question": "Как его расторгнуть?", "history": history})
        self.assertEqual(result, {"status": "ready", "search_request": search(), "message": None})
        self.assertEqual(len(mock.calls), 3)
        self.assertEqual(json.loads(mock.calls[0][1]["content"])["history"], history)
        self.assertEqual(json.loads(mock.calls[1][1]["content"])["question"], context()["standalone_query"])
        data = json.loads(mock.calls[2][1]["content"])
        self.assertEqual(data["history"], history)
        self.assertEqual(data["standalone_query"], context()["standalone_query"])
        self.assertIn("не являются источником права", mock.calls[0][0]["content"])

    async def test_clarification_stops_before_classification(self):
        """Уточняющий вопрос возвращается после первого этапа."""
        mock = MockLLM({"standalone_query": None, "clarification_question": "Что вы хотите изменить?"})
        result = await MultiStageQueryProcessingService(mock).process_query({"question": "Можно иначе?"})
        self.assertEqual(result, {"status": "clarification_required", "search_request": None,
                                  "message": "Что вы хотите изменить?"})
        self.assertEqual(len(mock.calls), 1)

    async def test_out_of_scope_stops_before_search(self):
        """Непрофильный вопрос классифицируется без подготовки поиска."""
        mock = MockLLM(context("Как сварить суп?"), {"is_legal": False, "message": "Вопрос не юридический."})
        result = await MultiStageQueryProcessingService(mock).process_query({"question": "Как сварить суп?"})
        self.assertEqual(result, {"status": "out_of_scope", "search_request": None,
                                  "message": "Вопрос не юридический."})
        self.assertEqual(len(mock.calls), 2)

    async def test_single_retry_is_shared_between_stages(self):
        """Один повтор на весь запрос расходуется на первом некорректном этапе."""
        mock = MockLLM("bad", context(), {"is_legal": "yes", "message": None})
        with self.assertRaises(InvalidLLMResponse):
            await MultiStageQueryProcessingService(mock).process_query({"question": "Вопрос"})
        self.assertEqual(len(mock.calls), 3)
        self.assertEqual(len(mock.calls[1]), 3)
        self.assertEqual(len(mock.calls[2]), 2)

    async def test_retry_can_be_used_at_later_stage(self):
        """При ошибке позднего этапа исправленный ответ принимается."""
        mock = MockLLM(context(), {"is_legal": True, "message": None}, "{}", search())
        result = await MultiStageQueryProcessingService(mock).process_query({"question": "Вопрос"})
        self.assertEqual(result["status"], "ready")
        self.assertEqual(len(mock.calls), 4)

    async def test_invalid_stage_contracts_are_rejected(self):
        """Некорректные поля и несогласованные результаты отклоняются."""
        bad = [
            ({"standalone_query": "", "clarification_question": None},),
            ({"standalone_query": "Вопрос", "clarification_question": "Какой?"},),
            (context(), {"is_legal": False, "message": None}),
            (context(), {"is_legal": True, "message": "Нет"}),
        ]
        for prefix in bad:
            with self.subTest(prefix=prefix):
                mock = MockLLM(*prefix, "{}")
                with self.assertRaises(InvalidLLMResponse):
                    await MultiStageQueryProcessingService(mock).process_query({"question": "Вопрос"})

    async def test_provider_errors_are_not_retried(self):
        """Таймауты и ошибки API сразу передаются вызывающей стороне."""
        for error in (LLMError("failure"), LLMTimeoutError("timeout")):
            mock = MockLLM(context(), error)
            with self.subTest(error=error), self.assertRaises(type(error)):
                await MultiStageQueryProcessingService(mock).process_query({"question": "Вопрос"})
            self.assertEqual(len(mock.calls), 2)

    async def test_retry_budget_is_per_request(self):
        """Параллельные запросы не разделяют лимит повторов."""
        class IndependentMock:
            async def complete(self, messages):
                """Заставляет каждый запрос повторить первый этап один раз."""
                if len(messages) == 2:
                    await asyncio.sleep(0)
                    return "bad"
                return json.dumps({"standalone_query": None, "clarification_question": "Что вы имеете в виду?"})
        service = MultiStageQueryProcessingService(IndependentMock())
        results = await asyncio.gather(*(service.process_query({"question": "А это?"}) for _ in range(2)))
        self.assertEqual([result["status"] for result in results], ["clarification_required"] * 2)


class MultiStageHttpTests(unittest.TestCase):
    def test_multi_mode_http(self):
        """HTTP-адаптер выбирает многошаговый сервис и сохраняет формат ответа."""
        env = {"LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "placeholder", "LLM_MODEL": "model",
               "QUERY_PROCESSING_MODE": "multi"}
        responses = [json.dumps(context()), json.dumps({"is_legal": True, "message": None}), json.dumps(search())]
        with patch.dict(os.environ, env, clear=True), patch("query_processing.llm_client.OpenAIClient.complete", side_effect=responses) as complete:
            with TestClient(app) as client:
                self.assertIsInstance(app.state.query_service, MultiStageQueryProcessingService)
                response = client.post("/query/process", json={"question": "Как расторгнуть договор?"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["search_request"], search())
                self.assertEqual(complete.await_count, 3)

    def test_invalid_mode_fails_at_startup(self):
        """Ошибка режима обнаруживается до обслуживания HTTP-запросов."""
        env = {"LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "placeholder", "LLM_MODEL": "model",
               "QUERY_PROCESSING_MODE": "invalid"}
        with patch.dict(os.environ, env, clear=True), self.assertRaises(ConfigurationError):
            with TestClient(app):
                pass
