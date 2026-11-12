"""Проверки сервиса и HTTP-адаптера без настоящих запросов к LLM."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import openai
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.main import app
from query_processing.config import ConfigurationError, LLMSettings
from query_processing.contracts import QueryRequest
from query_processing.llm_client import LLMError, LLMTimeoutError, OpenAIClient
from query_processing.service import QueryProcessingService
from query_processing.validator import InvalidLLMResponse, validate_response


def ready(**overrides):
    """Создаёт пример корректного результата для тестов."""
    search = {
        "standalone_query": "Как расторгнуть договор?",
        "search_queries": ["расторжение договора"],
        "filters": {"document_number": None, "document_type": "договор", "event_date": None},
    }
    search.update(overrides)
    return {"status": "ready", "search_request": search, "message": None}


class MockLLM:
    def __init__(self, *responses):
        """Сохраняет последовательность ответов и вызовы."""
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages):
        """Возвращает следующий подготовленный ответ или ошибку."""
        self.calls.append(messages)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_legal_question_without_document_number(self):
        """Понятный общий юридический вопрос направляется в поиск."""
        mock = MockLLM(ready())
        result = await QueryProcessingService(mock).process_query({"question": "Как расторгнуть договор?"})
        self.assertEqual(result, ready())
        self.assertEqual(len(mock.calls), 1)
        self.assertEqual(json.loads(mock.calls[0][1]["content"])["history"], [])

    async def test_out_of_scope(self):
        """Непрофильный вопрос не создаёт поисковый запрос."""
        response = {"status": "out_of_scope", "search_request": None, "message": "Вопрос не юридический."}
        result = await QueryProcessingService(MockLLM(response)).process_query({"question": "Как сварить суп?"})
        self.assertEqual(result, response)

    async def test_history_is_passed_and_context_is_retained(self):
        """История целиком передаётся модели для восстановления контекста."""
        history = [{"role": "user", "content": "Условия договора №123?"},
                   {"role": "assistant", "content": "Условия уведомления."}]
        response = ready(standalone_query="Как расторгнуть договор №123?",
                         filters={"document_number": "123", "document_type": "договор", "event_date": None})
        mock = MockLLM(response)
        result = await QueryProcessingService(mock).process_query({"question": "А как его расторгнуть?", "history": history})
        self.assertEqual(json.loads(mock.calls[0][1]["content"])["history"], history)
        self.assertEqual(result, response)
        self.assertIn("не является источником", mock.calls[0][0]["content"].replace("\n", " "))

    async def test_ambiguous_question(self):
        """Неоднозначный вопрос без контекста возвращает уточнение."""
        response = {"status": "clarification_required", "search_request": None, "message": "Что вы имеете в виду?"}
        result = await QueryProcessingService(MockLLM(response)).process_query({"question": "А можно иначе?"})
        self.assertEqual(result, response)

    async def test_number_date_and_multiple_queries(self):
        """Номер, дата и три поисковые строки проходят без изменения."""
        response = ready(search_queries=["договор 123", "расторжение 123", "уведомление 123"],
                         filters={"document_number": "123", "document_type": "договор", "event_date": "2025-04-15"})
        result = await QueryProcessingService(MockLLM(response)).process_query({"question": "Договор №123 от 15.04.2025"})
        self.assertEqual(result, response)

    async def test_invalid_json_retried_once(self):
        """После одного неверного ответа сервис принимает исправленный JSON."""
        mock = MockLLM("invalid json", ready())
        self.assertEqual(await QueryProcessingService(mock).process_query({"question": "Как расторгнуть договор?"}), ready())
        self.assertEqual(len(mock.calls), 2)
        self.assertEqual(len(mock.calls[1]), 3)

    async def test_invalid_json_twice_fails(self):
        """Два некорректных ответа завершаются ошибкой без третьего вызова."""
        mock = MockLLM("{", "{}")
        with self.assertRaises(InvalidLLMResponse):
            await QueryProcessingService(mock).process_query({"question": "Как расторгнуть договор?"})
        self.assertEqual(len(mock.calls), 2)

    async def test_model_errors_do_not_trigger_retry(self):
        """Ошибки поставщика и таймауты передаются без повторов."""
        for error in [LLMError("failed"), LLMTimeoutError("timed out")]:
            with self.subTest(error=error):
                mock = MockLLM(error)
                with self.assertRaises(type(error)):
                    await QueryProcessingService(mock).process_query({"question": "Вопрос"})
                self.assertEqual(len(mock.calls), 1)

    async def test_direct_import_does_not_load_api(self):
        """Сервис можно импортировать и вызвать в отдельном процессе без API."""
        import subprocess
        import sys
        code = '''import asyncio, json, sys
from query_processing.service import QueryProcessingService
class Fake:
 async def complete(self, messages):
  return json.dumps({"status":"out_of_scope","search_request":None,"message":"Нет"})
result = asyncio.run(QueryProcessingService(Fake()).process_query({"question":"Привет"}))
assert result["status"] == "out_of_scope" and "fastapi" not in sys.modules
'''
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ValidationTests(unittest.TestCase):
    def test_invalid_results(self):
        """Отклоняет неверные статусы, даты, строки и несовместимые поля."""
        bad_values = [
            {"status": "ready", "search_request": None, "message": None},
            {"status": "out_of_scope", "search_request": None, "message": " "},
            ready(search_queries=[]), ready(search_queries=["1", "2", "3", "4"]),
            ready(search_queries=[" "]),
            ready(filters={"document_number": None, "document_type": None, "event_date": "2025-02-30"}),
            ready(filters={"document_number": 123, "document_type": None, "event_date": None}),
        ]
        for value in bad_values:
            with self.subTest(value=value), self.assertRaises(InvalidLLMResponse):
                validate_response(json.dumps(value))

    def test_invalid_input(self):
        """Отклоняет пустой вопрос и неподдерживаемые роли истории."""
        for query in [{"question": " "}, {"question": "?", "history": [{"role": "system", "content": "x"}]}]:
            with self.subTest(query=query), self.assertRaises(ValidationError):
                QueryRequest.model_validate(query)

    def test_settings_validation_and_optional_parameters(self):
        """Проверяет настройки и разрешает отключить параметры генерации."""
        env = {"LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "secret", "LLM_MODEL": "model",
               "LLM_TEMPERATURE": "", "LLM_MAX_TOKENS": ""}
        with patch.dict(os.environ, env, clear=True):
            settings = LLMSettings.from_environment()
        self.assertIsNone(settings.temperature)
        self.assertIsNone(settings.max_tokens)
        self.assertNotIn("secret", repr(settings))
        with self.assertRaises(ConfigurationError):
            LLMSettings(base_url="invalid", api_key="secret", model="model")


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_optional_params_are_not_sent(self):
        """Не отправляет отключённые параметры генерации провайдеру."""
        settings = LLMSettings("https://example.com/v1", "secret", "model", None, None)
        client = OpenAIClient(settings)
        create = AsyncMock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))]))
        with patch.object(client._client.chat.completions, "create", create):
            self.assertEqual(await client.complete([{"role": "user", "content": "test"}]), "{}")
        self.assertEqual(set(create.await_args.kwargs), {"model", "messages"})
        await client.close()

    async def test_api_errors_are_sanitized(self):
        """Преобразует ошибки SDK и не раскрывает их исходный текст."""
        request = Mock()
        errors = [openai.APITimeoutError(request=request), openai.APIConnectionError(request=request)]
        client = OpenAIClient(LLMSettings("https://example.com/v1", "secret", "model"))
        for error, expected in zip(errors, [LLMTimeoutError, LLMError]):
            with self.subTest(error=error), patch.object(client._client.chat.completions, "create", AsyncMock(side_effect=error)):
                with self.assertRaises(expected) as captured:
                    await client.complete([])
                self.assertNotIn("secret", str(captured.exception))
        await client.close()


class HttpTests(unittest.TestCase):
    def test_http_contract_and_error_mapping(self):
        """Проверяет HTTP-контракт, входную валидацию и коды ошибок."""
        env = {"LLM_BASE_URL": "https://example.com/v1", "LLM_API_KEY": "secret", "LLM_MODEL": "model"}
        with patch.dict(os.environ, env, clear=True), TestClient(app) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ok"})
            app.state.query_service = QueryProcessingService(MockLLM(ready(), LLMTimeoutError(), LLMError(), "bad", "bad"))
            self.assertEqual(client.post("/query/process", json={"question": " "}).status_code, 422)
            response = client.post("/query/process", json={"question": "Как расторгнуть договор?"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), ready())
            self.assertEqual(client.post("/query/process", json={"question": "x"}).status_code, 504)
            self.assertEqual(client.post("/query/process", json={"question": "x"}).status_code, 502)
            self.assertEqual(client.post("/query/process", json={"question": "x"}).status_code, 502)


if __name__ == "__main__":
    unittest.main()
