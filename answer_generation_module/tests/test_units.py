"""Модульные проверки: числа, инъекции, разбор JSON, синхронный адаптер, HTTP."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from fastapi.testclient import TestClient

from answer_generation import SyncLLMAdapter, generate_answer_sync
from answer_generation.config import GenerationSettings
from answer_generation.evidence import infer_event_period, sanitize_text
from answer_generation.numbers import missing_facts, numeric_facts
from answer_generation.parsing import ParseError, coerce_sources, parse_draft
from answer_generation.service import AnswerGenerationService

from tests.fixtures import LEASE_RENT, MEMO, MockLLM, draft, request, verdicts


class NumbersTests(unittest.TestCase):
    def test_normalization(self):
        facts = numeric_facts("плата 450 000 руб.; пени 0,1%; договор на 12 млн; 01.08.2024; п. 3.1")
        self.assertTrue({"450000", "0.1", "12000000", "D:2024-08-01", "3.1"} <= facts)

    def test_words_and_digits_are_equal(self):
        self.assertEqual(missing_facts("не более 6 месяцев", "не может превышать шести месяцев"), [])
        self.assertEqual(missing_facts("в течение 45 дней", "в течение сорока пяти дней"), [])
        self.assertEqual(missing_facts("с 1 августа 2024 г.", "с 01.08.2024"), [])

    def test_invented_number_detected(self):
        self.assertEqual(missing_facts("срок оплаты — 90 дней", "в течение 30 (тридцати) календарных дней"), ["90"])

    def test_structural_refs_are_soft(self):
        self.assertEqual(missing_facts("ч. 5 ст. 70 ТК РФ", "ТК РФ, ст. 70 ..."), [])
        self.assertEqual(missing_facts("ст. 81 ТК РФ", "ТК РФ, ст. 70 ..."), ["81"])


class EvidenceUnitTests(unittest.TestCase):
    def test_sanitize_keeps_useful_text(self):
        text, removed = sanitize_text(MEMO["text"])
        self.assertTrue(removed)
        self.assertIn("не более 3 (трёх) рабочих дней", text)
        self.assertIn("Лебедев", text)
        self.assertNotIn("90", text)

    def test_sanitize_leaves_clean_text_untouched(self):
        text, removed = sanitize_text(LEASE_RENT["text"])
        self.assertFalse(removed)
        self.assertEqual(text, LEASE_RENT["text"])

    def test_event_period(self):
        self.assertEqual(infer_event_period("Подписан 01.08.2024, проверьте").start, date(2024, 8, 1))
        self.assertEqual(infer_event_period("в 2024 году").end, date(2024, 12, 31))
        self.assertIsNone(infer_event_period("договор ПТ-05/2023 и П-17/2024"))
        self.assertIsNone(infer_event_period("с 01.08.2024 по 01.03.2025"))


class ParsingTests(unittest.TestCase):
    def test_sources_coercion(self):
        self.assertEqual(coerce_sources([1, "2", "[3]", "фрагмент 4", 5.0, 1]), [1, 2, 3, 4, 5])

    def test_markers_stripped(self):
        parsed = parse_draft(draft(("Плата 450 000 рублей [1], [2].", "fact", [1])))
        self.assertEqual(parsed.sentences[0].text, "Плата 450 000 рублей.")

    def test_garbage_raises(self):
        with self.assertRaises(ParseError):
            parse_draft("не знаю")


class MainStyleClient:
    """Как llm_client.LLMClient из ветки main: синхронный chat(messages, temperature)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.lock = threading.Lock()

    def chat(self, messages, temperature=None):
        with self.lock:
            return self.replies.pop(0)


class SyncTests(unittest.TestCase):
    def test_sync_wrapper_with_main_style_client(self):
        client = MainStyleClient(draft(("Арендная плата — 450 000 рублей в месяц (п. 3.1).", "fact", [1])),
                                 verdicts(("s1", "supported", [1], "")))
        result = generate_answer_sync(request("Какая плата?", LEASE_RENT), SyncLLMAdapter(client))
        self.assertEqual(result["status"], "answered")

    def test_sync_wrapper_in_thread_pool(self):
        def solve(i):
            client = MainStyleClient(draft(("Арендная плата — 450 000 рублей в месяц (п. 3.1).", "fact", [1])),
                                     verdicts(("s1", "supported", [1], "")))
            return generate_answer_sync(request("Какая плата?", LEASE_RENT, rid=f"r{i}"),
                                        SyncLLMAdapter(client))["request_id"]

        with ThreadPoolExecutor(max_workers=4) as pool:
            self.assertEqual(sorted(pool.map(solve, range(8))), sorted(f"r{i}" for i in range(8)))


class ApiTests(unittest.TestCase):
    def setUp(self):
        from api.main import app
        self.app = app
        self.llm = MockLLM(draft(("Арендная плата — 450 000 рублей в месяц (п. 3.1).", "fact", [1])),
                           verdicts(("s1", "supported", [1], "")))
        app.state.answer_service = AnswerGenerationService(self.llm, GenerationSettings())
        self.client = TestClient(app)          # без with: lifespan с настоящим клиентом не запускается

    def test_generate_endpoint(self):
        response = self.client.post("/answer/generate", json=request("Какая плата?", LEASE_RENT))
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertNotIn("debug", body)

    def test_debug_flag(self):
        response = self.client.post("/answer/generate?debug=true", json=request("Какая плата?", LEASE_RENT))
        self.assertIn("debug", response.json())
        self.assertEqual(response.json()["debug"]["llm_calls"], 2)

    def test_invalid_request(self):
        response = self.client.post("/answer/generate", json={"request_id": "x", "question": "  "})
        self.assertEqual(response.status_code, 422)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
