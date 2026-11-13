"""Интеграционные и контрактные проверки сквозного юридического пайплайна."""

import asyncio
import csv
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from API import for_solutions
from answer_generation.models import Claim
from answer_generation.service import INSUFFICIENT, generate_verified, max_retries
from answer_generation.verification import source_errors
from answer_generation.verify import main as verify_main, verify_records
from data_processing.hybrid import FusedHit
from data_processing import config as search_config
from query_processing_module.query_processing.config import LLMSettings
from query_processing_module.query_processing.multi_stage import (
    CLASSIFICATION_PROMPT, CONTEXT_PROMPT, SEARCH_PROMPT,
)
from query_processing_module.query_processing.prompts import SYSTEM_PROMPT
from answer_generation.service import GENERATION_PROMPT
from answer_generation.verification import VERIFY_PROMPT
from rag_pipeline.corpus import Corpus
from rag_pipeline.retrieval import retrieve

ROOT = Path(__file__).resolve().parents[1]
DOC_NAME = "документы компании__Договор поставки П-17-2024.txt"
DOC_TEXT = ("ДОГОВОР ПОСТАВКИ № П-17/2024\n1. Условия\n"
            "1.1. Оплата производится в течение 10 дней.\n"
            "1.2. Товар передается по акту.")


def write_archive(directory: Path, text: str = DOC_TEXT) -> None:
    """Создаёт небольшой архив с воспроизводимым текстовым документом."""
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(directory / "knowledge-base.zip", "w") as archive:
        archive.writestr(DOC_NAME, text)


def ready_response() -> dict:
    """Возвращает подготовленный поисковый запрос по контракту модуля 2."""
    return {
        "status": "ready",
        "search_request": {
            "standalone_query": "Когда производится оплата?",
            "search_queries": ["оплата по договору", "срок платежа"],
            "filters": {"document_number": None, "document_type": None, "event_date": None},
        },
        "message": None,
    }


class QueueClient:
    """Возвращает заранее заданные ответы и сохраняет полученные промпты."""

    def __init__(self, *responses):
        """Подготавливает очередь ответов модели."""
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages):
        """Возвращает следующий ответ без сетевого обращения."""
        self.calls.append(messages)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return json.dumps(item, ensure_ascii=False)

    async def close(self):
        """Закрывает подменяемый клиент без сетевых ресурсов."""


def draft(chunk_id: str, text: str = "Оплата производится в течение 10 дней.") -> dict:
    """Формирует структурированный черновик с одним источником."""
    return {"status": "answered", "claims": [{"text": text, "chunk_ids": [chunk_id]}]}


def check(supported: bool, reason: str = "") -> dict:
    """Формирует результат независимой проверки одного утверждения."""
    return {"checks": [{"claim_index": 0, "supported": supported, "reason": reason}]}


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    """Проверяет существование ссылок, смысловую оценку и повторные попытки."""

    def setUp(self):
        """Создаёт компактный реестр подлинных источников."""
        self.source = {
            "chunk_id": "c1", "document_id": "d1", "document_title": "Договор № 1",
            "section": "п. 1.1", "text": "Оплата через 10 дней.",
            "citation": "Договор № 1, п. 1.1",
        }
        self.corpus = SimpleNamespace(version="v1", sources={"c1": self.source, "c2": {
            **self.source, "chunk_id": "c2", "section": "п. 1.2",
            "citation": "Договор № 1, п. 1.2",
        }})

    def test_unknown_not_sent_and_modified_sources(self):
        """Отклоняет несуществующую, не переданную и изменённую ссылку."""
        unknown = Claim(text="Факт", chunk_ids=["missing"])
        not_sent = Claim(text="Факт", chunk_ids=["c2"])
        valid = Claim(text="Факт", chunk_ids=["c1"])
        self.assertIn("неизвестный", source_errors([unknown], [self.source], self.corpus)[0])
        self.assertIn("не передан", source_errors([not_sent], [self.source], self.corpus)[0])
        modified = {**self.source, "text": "Подменённый текст"}
        self.assertIn("отличается", source_errors([valid], [modified], self.corpus)[0])
        embedded = Claim(text="Согласно пункту 99.1 срок составляет 10 дней", chunk_ids=["c1"])
        self.assertIn("ссылка внутри текста", source_errors([embedded], [self.source], self.corpus)[0])

    async def test_invalid_citation_regenerates_and_checks_semantics(self):
        """Передаёт ошибку генератору и выдаёт ответ только после проверки."""
        client = QueueClient(draft("missing"), draft("c1"), check(True))
        result = await generate_verified(client, "Когда оплата?", [self.source], self.corpus, retries=1)
        self.assertEqual(result["status"], "answered")
        self.assertIn("Договор № 1, п. 1.1", result["answer"])
        self.assertEqual(len(result["attempts"]), 2)
        retry_payload = json.loads(client.calls[1][1]["content"])
        self.assertIn("неизвестный", retry_payload["previous_errors"][0])
        self.assertEqual(len(client.calls), 3)

    async def test_missing_citation_reports_field_to_regeneration(self):
        """Передаёт модели конкретное поле, в котором отсутствует ссылка."""
        invalid = {"status": "answered", "claims": [{"text": "Факт", "chunk_ids": []}]}
        client = QueueClient(invalid, draft("c1"), check(True))
        result = await generate_verified(client, "Вопрос", [self.source], self.corpus, retries=1)
        self.assertEqual(result["status"], "answered")
        feedback = json.loads(client.calls[1][1]["content"])["previous_errors"]
        self.assertIn("chunk_ids", feedback[0])

    async def test_unsupported_claim_and_zero_retries_returns_refusal(self):
        """Не выдаёт утверждение, которое проверяющая модель не подтвердила."""
        client = QueueClient(draft("c1"), check(False, "срок не указан"))
        result = await generate_verified(client, "Когда оплата?", [self.source], self.corpus, retries=0)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["answer"], INSUFFICIENT)
        self.assertEqual(len(client.calls), 2)
        self.assertIn("срок не указан", result["verification"]["errors"][0])

    async def test_missing_checks_and_provider_failure_are_not_approved(self):
        """Отклоняет неполный ответ проверяющей модели и ошибку провайдера."""
        for second in ({"checks": []}, RuntimeError("provider unavailable")):
            client = QueueClient(draft("c1"), second)
            result = await generate_verified(client, "Вопрос", [self.source], self.corpus, retries=0)
            self.assertNotEqual(result["status"], "answered")
            self.assertEqual(result["answer"], INSUFFICIENT)

    async def test_separate_recheck_regenerates_invalid_reference(self):
        """Повторная проверка исправляет сохранённую неверную ссылку."""
        record = {
            "id": "q1", "question": "Когда оплата?", "corpus_version": "v1",
            "status": "answered", "answer": "Старый ответ", "claims": draft("missing")["claims"],
            "sources": [self.source], "search_request": ready_response()["search_request"],
        }
        client = QueueClient(draft("c1"), check(True))
        result = await verify_records([record], {"q1": "Старый ответ"}, self.corpus, client, None, 1)
        self.assertEqual(result[0]["status"], "answered")
        self.assertIn("Договор № 1, п. 1.1", result[0]["answer"])
        self.assertIn("неизвестный", result[0]["recheck_errors"][0])

    async def test_changed_corpus_retrieves_fresh_context(self):
        """При изменившемся корпусе не переиспользует старые фрагменты."""
        record = {
            "id": "q1", "question": "Когда оплата?", "corpus_version": "old",
            "status": "answered", "answer": "Ответ", "claims": draft("c1")["claims"],
            "sources": [self.source], "search_request": ready_response()["search_request"],
        }
        client = QueueClient(draft("c1"), check(True))
        with patch("answer_generation.verify.retrieve", return_value=[self.source]) as search:
            result = await verify_records([record], {"q1": "Ответ"}, self.corpus, client, object(), 0)
        search.assert_called_once()
        self.assertEqual(result[0]["corpus_version"], "v1")
        self.assertEqual(result[0]["status"], "answered")

    def test_retry_environment(self):
        """Интерпретирует значение окружения как число дополнительных попыток."""
        with patch.dict(os.environ, {"ANSWER_VERIFY_MAX_RETRIES": "3"}):
            self.assertEqual(max_retries(), 3)
        with patch.dict(os.environ, {"ANSWER_VERIFY_MAX_RETRIES": "-1"}):
            with self.assertRaises(ValueError):
                max_retries()


class FakeSearcher:
    """Возвращает фрагменты корпуса и фиксирует переданные формулировки."""

    def __init__(self, corpus):
        """Выбирает текстовый пункт договора как результат команды."""
        self.queries = []
        self.chunk = next(chunk for chunk in corpus.chunks if chunk.label == "п. 1.1")

    def search(self, query, *, final_k):
        """Возвращает результат в формате FusedHit поискового модуля."""
        self.queries.append(query)
        return [FusedHit(chunk=self.chunk, rrf_score=0.02, bm25_rank=1, bm25_score=8.0)]


class ScenarioClient:
    """Имитирует Query Processing, генератор и независимого проверяющего."""

    def __init__(self, settings):
        """Создаёт подменяемый клиент без сети."""
        self.settings = settings

    async def complete(self, messages):
        """Определяет этап по системному промпту и выдаёт его контракт."""
        prompt, payload = messages[0]["content"], json.loads(messages[1]["content"])
        if prompt == SYSTEM_PROMPT:
            result = ({"status": "out_of_scope", "message": "Вопрос не юридический", "search_request": None}
                      if "суп" in payload["question"] else ready_response())
        elif prompt == CONTEXT_PROMPT:
            result = {"standalone_query": payload["question"], "clarification_question": None}
        elif prompt == CLASSIFICATION_PROMPT:
            legal = "суп" not in payload["question"]
            result = {"is_legal": legal, "message": None if legal else "Вопрос не юридический"}
        elif prompt == SEARCH_PROMPT:
            result = ready_response()["search_request"]
        elif prompt == GENERATION_PROMPT:
            selected = next((source for source in payload["sources"] if "10 дней" in source["text"]),
                            payload["sources"][0])
            result = draft(selected["chunk_id"])
        elif prompt == VERIFY_PROMPT:
            result = check(True)
        else:
            raise AssertionError("Неизвестный промпт")
        return json.dumps(result, ensure_ascii=False)

    async def close(self):
        """Закрывает подменяемый клиент."""


class IntegrationTests(unittest.TestCase):
    """Проверяет CSV-оркестрацию и адаптер существующего поиска команды."""

    def test_retrieval_passes_variants_filters_and_preserves_provenance(self):
        """Вызывает поисковый модуль для всех формулировок и удаляет дубли."""
        with tempfile.TemporaryDirectory() as directory:
            write_archive(Path(directory))
            corpus = Corpus.from_directory(Path(directory))
        searcher = FakeSearcher(corpus)
        team = SimpleNamespace(searcher=searcher)
        request = ready_response()["search_request"]
        request["filters"]["document_number"] = "П-17/2024"
        request["filters"]["document_type"] = "contract"
        sources = retrieve(corpus, team, "Исходный вопрос", request)
        self.assertEqual(len(searcher.queries), 4)
        self.assertEqual(len([item for item in sources if item["selection_reason"] == "retrieved"]), 1)
        self.assertEqual(len(sources[0]["matched_queries"]), 4)
        self.assertEqual(sources[0]["matched_by"], ["bm25"])
        self.assertEqual(sources[0]["document_number"], "П-17/2024")
        self.assertIsNone(sources[0]["status"])

    def test_retrieval_keeps_team_supplied_related_supplement(self):
        """Не теряет найденное допсоглашение при фильтре основного договора."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_archive(root)
            with zipfile.ZipFile(root / "knowledge-base.zip", "a") as archive:
                archive.writestr(
                    "документы компании__Доп соглашение 1 к договору П-17-2024.txt",
                    "ДОПОЛНИТЕЛЬНОЕ СОГЛАШЕНИЕ № 1\n"
                    "к договору поставки № П-17/2024 от 12 февраля 2024 г.\n"
                    "1. Изложить пункт 1.1 договора в следующей редакции: оплата через 20 дней.",
                )
            corpus = Corpus.from_directory(root)
        supplement = next(chunk for chunk in corpus.chunks if chunk.kind == "доп.соглашение")
        parent = corpus.metas[supplement.doc_id].parent_contract
        self.assertEqual(parent, "П-17/2024")
        hit = FusedHit(chunk=supplement, rrf_score=0.02, bm25_rank=1)
        team = SimpleNamespace(searcher=SimpleNamespace(search=lambda *args, **kwargs: [hit]))
        request = ready_response()["search_request"]
        request["filters"] = {"document_number": "П-17/2024", "document_type": "contract", "event_date": None}
        found = retrieve(corpus, team, "Вопрос", request)
        self.assertEqual(found[0]["parent_contract"], "П-17/2024")
        self.assertEqual(found[0]["document_type"], "доп.соглашение")

    def test_retrieval_includes_candidates_from_distinct_formulations(self):
        """Не вытесняет поздние формулировки полным списком первого запроса."""
        with tempfile.TemporaryDirectory() as directory:
            write_archive(Path(directory))
            corpus = Corpus.from_directory(Path(directory))
        def search(query, *, final_k):
            """Имитирует разные списки результатов поискового модуля."""
            chunks = corpus.chunks[:2] if query == "Первый" else [corpus.chunks[2]]
            return [FusedHit(chunk=chunk, rrf_score=0.02, bm25_rank=i + 1)
                    for i, chunk in enumerate(chunks)]
        team = SimpleNamespace(searcher=SimpleNamespace(search=search))
        request = {"standalone_query": "Второй", "search_queries": [], "filters": {}}
        result = retrieve(corpus, team, "Первый", request, limit=2)
        self.assertEqual([item["chunk_id"] for item in result],
                         [corpus.chunks[2].chunk_id, corpus.chunks[0].chunk_id])

    def test_verify_cli_writes_repaired_copies(self):
        """Отдельная команда исправляет ошибочную ссылку и сохраняет копии файлов."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_archive(data)
            corpus = Corpus.from_directory(data)
            source = next(item for item in corpus.sources.values() if item["section"] == "п. 1.1")
            record = {"id": "q1", "question": "Когда оплата?", "corpus_version": corpus.version,
                      "status": "answered", "answer": "Старый ответ", "claims": draft("missing")["claims"],
                      "sources": [source], "search_request": ready_response()["search_request"]}
            trace, submission = root / "trace.jsonl", root / "submission.csv"
            trace.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            with submission.open("w", encoding="utf-8", newline="") as target:
                writer = csv.writer(target)
                writer.writerow(["id", "answer"])
                writer.writerow(["q1", "Старый ответ"])
            output, output_trace = root / "verified.csv", root / "verified.jsonl"
            args = ["verify", "--trace", str(trace), "--submission", str(submission),
                    "--out", str(output), "--out-trace", str(output_trace)]
            settings = LLMSettings("https://example.invalid/v1", "fake-key", "fake-model")
            client = QueueClient(draft(source["chunk_id"]), check(True))
            team = SimpleNamespace(close=lambda: None)
            with (patch.object(sys, "argv", args), patch("answer_generation.verify.OpenAIClient", return_value=client),
                  patch.object(LLMSettings, "from_environment", return_value=settings),
                  patch("answer_generation.verify.Corpus.from_directory", return_value=corpus),
                  patch("answer_generation.verify.create_search", return_value=team)):
                verify_main()
            with output.open(encoding="utf-8", newline="") as source_file:
                result = list(csv.DictReader(source_file))
            self.assertIn("Договор поставки № П-17/2024, п. 1.1", result[0]["answer"])
            self.assertIn("Старый ответ", submission.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(output_trace.read_text(encoding="utf-8"))["status"], "answered")

    def test_solution_outputs_verified_answers_and_trace_in_both_modes(self):
        """Запускает обе реализации Query Processing с настоящим CSV и ZIP."""
        for mode in ("single", "multi"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / "data"
                write_archive(data)
                with (data / "questions.csv").open("w", encoding="utf-8", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(["id", "question"])
                    writer.writerows([["1", "Когда оплата?"], ["2", "Как сварить суп?"]])
                settings = LLMSettings("https://example.invalid/v1", "fake-key", "fake-model")
                environment = {"DATA_DIR": str(data), "ARTIFACTS_DIR": str(root / "artifacts"),
                               "QUERY_PROCESSING_MODE": mode, "MAX_WORKERS": "1"}
                with (patch.dict(os.environ, environment), patch("API.for_solutions.Path.cwd", return_value=root),
                      patch("API.for_solutions.OpenAIClient", ScenarioClient),
                      patch.object(LLMSettings, "from_environment", return_value=settings),
                      patch("API.for_solutions.create_search",
                            side_effect=lambda corpus, _: SimpleNamespace(
                                searcher=FakeSearcher(corpus), close=lambda: None))):
                    for_solutions.main()
                with (root / "submission.csv").open(encoding="utf-8", newline="") as source:
                    rows = list(csv.DictReader(source))
                records = [json.loads(line) for line in
                           (root / "artifacts" / "answer_trace.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([row["id"] for row in rows], ["1", "2"])
                self.assertIn("Договор поставки № П-17/2024, п. 1.1", rows[0]["answer"])
                self.assertEqual(rows[1]["answer"], "Вопрос не юридический")
                self.assertEqual(records[0]["verification"]["status"], "passed")
                self.assertEqual(records[1]["status"], "out_of_scope")
                self.assertEqual(records[0]["sources"][0]["parent_contract"], None)


class EmbeddingHandler(BaseHTTPRequestHandler):
    """Имитирует совместимый API эмбеддингов для Makefile."""

    calls = 0

    def do_POST(self):
        """Возвращает одинаковые векторы для текстов входного батча."""
        size = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(size))
        type(self).calls += 1
        result = {"data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(payload["input"])]}
        encoded = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format, *args):
        """Не выводит локальные запросы тестового сервера."""


class MockPlatformHandler(EmbeddingHandler):
    """Имитирует одновременно API модели и эмбеддингов платформы."""

    def do_POST(self):
        """Обрабатывает chat completions или делегирует подсчёт эмбеддингов."""
        if self.path.endswith("/embeddings"):
            return super().do_POST()
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = payload["messages"][0]["content"]
        data = json.loads(payload["messages"][1]["content"])
        if prompt == SYSTEM_PROMPT:
            result = ready_response()
        elif prompt == GENERATION_PROMPT:
            source = next(item for item in data["sources"] if "10 дней" in item["text"])
            result = draft(source["chunk_id"])
        elif prompt == VERIFY_PROMPT:
            result = check(True)
        else:
            raise AssertionError("Неизвестный промпт")
        body = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                "model": "mock", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": json.dumps(result, ensure_ascii=False)}}]}
        encoded = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class IndexCommandTests(unittest.TestCase):
    """Проверяет Makefile и переиспользование индекса команды."""

    def test_make_commands_build_reuse_and_rebuild_index(self):
        """Проверяет чанки, кэш, принудительное и обусловленное данными обновление."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        EmbeddingHandler.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_archive(data)
            env = {**os.environ, "DATA_DIR": str(data), "ARTIFACTS_DIR": str(root / "artifacts"),
                   "EMB_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                   "EMB_API_KEY": "fake", "EMB_MODEL": "fake-embeddings"}
            def run_make(target):
                """Запускает одну цель Makefile на тестовом архиве."""
                subprocess.run(["make", target, f"PYTHON={sys.executable}"], cwd=ROOT,
                               env=env, check=True, capture_output=True, text=True)
            run_make("chunks")
            self.assertTrue((root / "artifacts" / "chunks.jsonl").is_file())
            self.assertEqual(EmbeddingHandler.calls, 0)
            run_make("index")
            self.assertTrue((root / "artifacts" / "vector_index.npz").is_file())
            initial = EmbeddingHandler.calls
            self.assertGreater(initial, 0)
            run_make("index")
            self.assertEqual(EmbeddingHandler.calls, initial)
            run_make("reindex")
            self.assertGreater(EmbeddingHandler.calls, initial)
            previous = EmbeddingHandler.calls
            write_archive(data, DOC_TEXT.replace("10 дней", "20 дней"))
            run_make("index")
            self.assertGreater(EmbeddingHandler.calls, previous)

    def test_solution_calls_actual_team_search(self):
        """Прогоняет реальный поиск команды с локальной подменой эмбеддингов и LLM."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), EmbeddingHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        EmbeddingHandler.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_archive(data)
            (data / "questions.csv").write_text("id,question\n1,Когда оплата?\n", encoding="utf-8")
            settings = LLMSettings("https://example.invalid/v1", "fake-key", "fake-model")
            environment = {"DATA_DIR": str(data), "ARTIFACTS_DIR": str(root / "artifacts"),
                           "QUERY_PROCESSING_MODE": "single", "MAX_WORKERS": "1"}
            with (patch.dict(os.environ, environment), patch("API.for_solutions.Path.cwd", return_value=root),
                  patch("API.for_solutions.OpenAIClient", ScenarioClient),
                  patch.object(LLMSettings, "from_environment", return_value=settings),
                  patch.object(search_config, "EMB_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"),
                  patch.object(search_config, "EMB_API_KEY", "fake"),
                  patch.object(search_config, "EMB_MODEL", "fake-embeddings"),
                  patch.object(search_config, "RERANK_MODE", "none")):
                for_solutions.main()
            record = json.loads((root / "artifacts" / "answer_trace.jsonl").read_text(encoding="utf-8"))
            selected_id = record["claims"][0]["chunk_ids"][0]
            selected = next(source for source in record["sources"] if source["chunk_id"] == selected_id)
            self.assertIn("10 дней", selected["text"])
            self.assertEqual(record["verification"]["status"], "passed")
            self.assertGreater(EmbeddingHandler.calls, 1)

    def test_solution_entrypoint_against_mock_platform(self):
        """Запускает настоящий solution.py с локальным OpenAI-совместимым API."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), MockPlatformHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            write_archive(data)
            (data / "questions.csv").write_text("id,question\n1,Когда оплата?\n", encoding="utf-8")
            base = f"http://127.0.0.1:{server.server_port}/v1"
            env = {**os.environ, "DATA_DIR": str(data), "ARTIFACTS_DIR": str(root / "artifacts"),
                   "LLM_BASE_URL": base, "LLM_API_KEY": "fake-key", "LLM_MODEL": "mock",
                   "EMB_BASE_URL": base, "EMB_API_KEY": "fake-key", "EMB_MODEL": "mock-embedding",
                   "RERANK_MODE": "none", "MAX_WORKERS": "1", "QUERY_PROCESSING_MODE": "single"}
            subprocess.run([sys.executable, str(ROOT / "solution.py")], cwd=root, env=env,
                           check=True, capture_output=True, text=True, timeout=30)
            with (root / "submission.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            trace = json.loads((root / "artifacts" / "answer_trace.jsonl").read_text(encoding="utf-8"))
            self.assertIn("Договор поставки № П-17/2024, п. 1.1", rows[0]["answer"])
            self.assertEqual(trace["verification"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
