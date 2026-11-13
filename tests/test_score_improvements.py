"""Регрессии полноты контекста, подтверждения ответа и устойчивости API."""

import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import openai

from API.backend_api import prepare_query
from API.for_solutions import QuestionRow, _process_rows, _early_result, _run_row
from answer_generation.context import model_sources
from answer_generation.models import Claim
from answer_generation.service import generate_verified, check_claims
from answer_generation.verification import source_errors
from data_processing import textproc
from data_processing.chunking import DocMeta, build_citation_title
from data_processing.hybrid import HybridSearcher
from data_processing.search import build_index
from data_processing.vector_index import VectorIndex
from query_processing_module.query_processing.config import LLMSettings
from query_processing_module.query_processing.json_utils import json_object
from query_processing_module.query_processing.llm_client import OpenAIClient, LLMError
from rag_pipeline.corpus import Corpus
from rag_pipeline.editions import question_period, applicability
from rag_pipeline.indexing import TeamSearch
from rag_pipeline.output import write_submission
from rag_pipeline.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]


class QueueClient:
    """Детерминированный LLM для проверки действительно переданных сообщений."""

    def __init__(self, *responses):
        """Сохраняет ответы, включая строки с Markdown-ограждением."""
        self.responses, self.calls = list(responses), []

    async def complete(self, messages):
        """Возвращает очередной ответ и запоминает контекст модели."""
        self.calls.append(messages)
        value = self.responses.pop(0)
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)

    async def close(self):
        """Поддерживает тот же жизненный цикл, что сетевой клиент."""


class InputAndEvidenceTests(unittest.TestCase):
    """Проверяет новые правила на реальной базе и независимых примерах."""

    @classmethod
    def setUpClass(cls):
        """Загружает тот же корпус, который будет подготовлен на платформе."""
        cls.corpus = Corpus.from_directory(ROOT / "data")

    def test_csv_dialogue_separates_last_user_turn_and_history(self):
        """Гарантия из прошлого ответа не подменяет последний вопрос о приёмке."""
        request = prepare_query("Пользователь: Поставщик ООО «Альфа-Трейд».\n"
                                "Ассистент: Гарантия 90 лет.\nПользователь: А срок приёмки?")
        self.assertEqual(request.question, "А срок приёмки?")
        self.assertEqual([item.role for item in request.history], ["user", "assistant"])
        self.assertIn("90 лет", request.history[1].content)
        self.assertEqual(prepare_query("  ДОГОВОР № А-08/2024? ").question, "ДОГОВОР № А-08/2024?")

    def test_inflected_legal_terms_have_same_stem(self):
        """Падежные формы больше не разводят юридические термины в разные основы."""
        for words in ("согласование согласованию согласования",
                      "одобрение одобрению одобрения", "начисление начислению начисления"):
            self.assertEqual(len(set(textproc.tokenize(words))), 1, words)

    def test_regulation_citation_does_not_invent_company_or_document_title(self):
        """Новый регламент не получает название организации из другого документа."""
        meta = DocMeta("r", "документы компании", "регламент", "", edition="01.02.2024")
        title = build_citation_title(meta, ["РЕГЛАМЕНТ УЧЁТА ООО «ВЕКТОР»"])
        self.assertIn("Регламент учёта ООО «Вектор»", title)
        self.assertNotIn("Меридиан", title)

    def test_historical_period_is_not_the_export_date(self):
        """Август 2024 выбирает старую редакцию до явной отмены, а не новую."""
        period = question_period("Договор заключён в августе 2024 года")
        self.assertEqual(period, ("2024-08-01", "2024-08-31"))
        sources = self.corpus.sources.values()
        old = next(item for item in sources if item["edition"] == "10.01.2023")
        new = next(item for item in sources if item["edition"] == "01.03.2025")
        self.assertEqual(applicability(old, period), "dated_edition_before_repeal")
        self.assertEqual(applicability(new, period), "outside_event_period")
        self.assertEqual(question_period("в марте 2024 года"), ("2024-03-01", "2024-03-31"))
        self.assertEqual(question_period("в мае 2024 года"), ("2024-05-01", "2024-05-31"))
        law = next(item for item in sources if item.get("valid_as_of"))
        self.assertIsNone(law["effective_from"])

    def test_model_context_removes_instructions_but_preserves_archive_rule(self):
        """Источник содержит трёхдневный срок без внедрённого ответа про 90 дней."""
        source = next(item for item in self.corpus.sources.values() if item["has_injection"])
        presented = model_sources([source])[0]
        self.assertEqual(presented["chunk_id"], "S1")
        self.assertNotIn("90 календарных", presented["text"])
        self.assertIn("не более 3", presented["text"])
        self.assertIn("90 календарных", source["text"])

    def test_bm25_fallback_keeps_more_than_five_candidates(self):
        """Отсутствие векторов не применяет глубину слияния к чистой лексике."""
        corpus = self.corpus
        searcher = HybridSearcher(build_index(corpus.chunks),
                                  VectorIndex(corpus.chunks, np.zeros((len(corpus.chunks), 1))))
        self.assertGreater(len(searcher.search("договор", final_k=15, use_rerank=False)), 5)

    def test_query_embeddings_batched_and_rerank_called_once(self):
        """Несколько формулировок используют один батч эмбеддингов и общий реранж."""
        corpus = self.corpus
        client = SimpleNamespace(embed=lambda texts: [[1.0] for _ in texts])
        searcher = HybridSearcher(build_index(corpus.chunks),
                                  VectorIndex(corpus.chunks, np.ones((len(corpus.chunks), 1))), client)
        request = {"standalone_query": "срок оплаты", "search_queries": ["платёж", "предоплата"], "filters": {}}
        with patch.object(client, "embed", wraps=client.embed) as embed, \
                patch.object(searcher, "rerank_candidates", side_effect=lambda _, hits: hits) as rerank:
            retrieve(corpus, TeamSearch(searcher, client), "срок оплаты", request, limit=6)
        embed.assert_called_once()
        self.assertEqual(len(embed.call_args.args[0]), 3)
        rerank.assert_called_once()

    def test_demo_evidence_covered_without_reading_reference_answers_at_runtime(self):
        """Все размеченные положения демо попадают в ограниченный контекст без сети."""
        from tools.benchmark_retrieval import coverage
        corpus = self.corpus
        team = TeamSearch(HybridSearcher(build_index(corpus.chunks),
                          VectorIndex(corpus.chunks, np.zeros((len(corpus.chunks), 1)))), None)
        with (ROOT / "data/demo_questions.csv").open(encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
        records = [{"id": row["id"], "sources": retrieve(corpus, team, row["question"], {
            "standalone_query": row["question"], "search_queries": [], "filters": {}})} for row in rows]
        expected = json.loads((ROOT / "tests/fixtures/demo_evidence.json").read_text())
        report = coverage(records, expected)
        self.assertEqual(report["evidence_found"], report["evidence_total"])


class GenerationAndRuntimeTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет подтверждение, восстановление ID и сохранение частичного прогона."""

    def setUp(self):
        """Создаёт независимый источник с короткими идентификаторами."""
        self.source = {"chunk_id": "real-id-42", "document_id": "doc", "document_title": "Договор № 42",
                       "section": "п. 1.1", "text": "Оплата через 10 дней.", "citation": "Договор № 42, п. 1.1"}
        self.corpus = SimpleNamespace(version="v1", sources={"real-id-42": self.source})

    async def test_fenced_json_alias_and_previous_draft_survive_regeneration(self):
        """Ошибка проверки относится к видимому модели прошлому черновику."""
        draft = {"status": "answered", "claims": [{"text": "Оплата через 10 дней.", "chunk_ids": ["S1"]}]}
        client = QueueClient(chr(96) * 3 + "json\n" + json.dumps(draft) + "\n" + chr(96) * 3,
                             {"checks": [{"claim_index": 0, "supported": False, "reason": "уточни формулировку"}]},
                             draft, {"checks": [{"claim_index": 0, "supported": True, "reason": ""}]})
        result = await generate_verified(client, "Когда оплата?", [self.source], self.corpus, retries=1)
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["claims"][0]["chunk_ids"], ["real-id-42"])
        payload = json.loads(client.calls[2][1]["content"])
        self.assertEqual(payload["previous_draft"][0]["chunk_ids"], ["S1"])
        self.assertIn("Оплата", payload["previous_draft"][0]["text"])

    async def test_recommended_word_count_does_not_discard_verified_answer(self):
        """Регламент допускает подробный ответ короче 10 000 символов."""
        claim = Claim(text=" ".join(["слово"] * 160), chunk_ids=["real-id-42"])
        client = QueueClient({"checks": [{"claim_index": 0, "supported": True, "reason": ""}]})
        answer, errors = await check_claims(client, "Вопрос", [claim], [self.source], self.corpus, None)
        self.assertFalse(errors)
        self.assertGreater(len(answer.split()), 150)

    def test_inline_reference_must_occur_in_cited_evidence(self):
        """Правильный номер допускается, придуманный номер блокируется до LLM."""
        valid = Claim(text="В пункте 1.1 предусмотрен срок оплаты.", chunk_ids=["real-id-42"])
        invalid = Claim(text="В пункте 99.1 предусмотрен срок оплаты.", chunk_ids=["real-id-42"])
        self.assertEqual(source_errors([valid], [self.source], self.corpus), [])
        self.assertTrue(source_errors([invalid], [self.source], self.corpus))
        with self.assertRaises(ValueError):
            json_object('{"status":"answered","status":"insufficient"}')

    async def test_transient_api_failure_retries_and_auth_failure_does_not(self):
        """429 восстанавливается, а неверный ключ не повторяется и не раскрывается."""
        settings = LLMSettings("https://test.invalid/v1", "private-token", "test")
        client = OpenAIClient(settings)
        self.addAsyncCleanup(client.close)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        request = httpx.Request("POST", "https://test.invalid/v1/chat/completions")
        retry = openai.RateLimitError("limited", response=httpx.Response(429, request=request), body=None)
        create = AsyncMock(side_effect=[retry, response])
        with patch.object(client._client.chat.completions, "create", create), patch("asyncio.sleep", new=AsyncMock()):
            self.assertEqual(await client.complete([]), "ok")
        self.assertEqual(create.await_count, 2)
        auth = openai.AuthenticationError("private-token", response=httpx.Response(401, request=request), body=None)
        create = AsyncMock(side_effect=auth)
        with patch.object(client._client.chat.completions, "create", create):
            with self.assertRaises(LLMError) as caught:
                await client.complete([])
        self.assertEqual(create.await_count, 1)
        self.assertNotIn("private-token", str(caught.exception))

    async def test_timeout_preserves_completed_csv_rows_and_all_ids(self):
        """Прерывание медленного вопроса оставляет уже готовый ответ в CSV."""
        rows = [QuestionRow(str(index), prepare_query("Вопрос"), "Вопрос") for index in range(2)]
        async def run_row(row, *args):
            """Завершает первый вопрос и имитирует задержку второго."""
            await asyncio.sleep(0 if row.row_id == "0" else 1)
            return {"id": row.row_id, **_early_result("answered", "Готовый ответ")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            settings = LLMSettings("https://test.invalid/v1", "key", "model")
            with patch("API.for_solutions.OpenAIClient", return_value=QueueClient()), \
                    patch("API.for_solutions._run_row", side_effect=run_row):
                records = await _process_rows(rows, settings, object, self.corpus, object(), 0,
                                             checkpoint=lambda items: write_submission(items, path), time_budget=0.02)
            with path.open() as source:
                saved = list(csv.DictReader(source))
        self.assertEqual([row["id"] for row in saved], ["0", "1"])
        self.assertEqual(saved[0]["answer"], "Готовый ответ")
        self.assertEqual(records[1]["verification"]["errors"], ["time_budget_exhausted"])
