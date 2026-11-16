"""Проверка отправляемого ZIP в каталоге без дерева исходного проекта."""

import csv
import hashlib
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

from answer_generation.service import GENERATION_PROMPT
from answer_generation.verification import VERIFY_PROMPT
from query_processing_module.query_processing.prompts import SYSTEM_PROMPT
from platform_solution.solution import SYSTEM_PROMPT as PLATFORM_PROMPT
from tools.build_platform_submission import build_archive, discover_modules


class PlatformAPI(BaseHTTPRequestHandler):
    """Подмена платформенного прокси с настоящим HTTP-контрактом."""

    fail_primary = False

    def do_POST(self):
        """Возвращает эмбеддинги или структурированный ответ текущего этапа."""
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            result = {"data": [{"index": index, "embedding": [1.0, 0.0]}
                               for index, _ in enumerate(data["input"])]}
        else:
            prompt = data["messages"][0]["content"]
            if prompt.startswith(PLATFORM_PROMPT) and self.fail_primary:
                self.send_error(400, "primary unavailable")
                return
            if prompt.startswith(PLATFORM_PROMPT):
                content = "ОТВЕТ: Оплата через 10 дней (п. 1.1)."
                payload = {}
            else:
                payload = json.loads(data["messages"][1]["content"])
            if prompt.startswith(PLATFORM_PROMPT):
                pass
            elif prompt == SYSTEM_PROMPT:
                content = {"status": "ready", "message": None, "search_request": {
                    "standalone_query": payload["question"], "search_queries": ["срок оплаты"],
                    "filters": {"document_number": None, "document_type": None, "event_date": None}}}
            elif prompt == GENERATION_PROMPT:
                source = next(source for source in payload["sources"] if "10 дней" in source["text"])
                content = {"status": "answered", "claims": [{"text": "Оплата через 10 дней.",
                            "chunk_ids": [source["chunk_id"]]}]}
            elif prompt == VERIFY_PROMPT:
                content = {"checks": [{"claim_index": index, "supported": True, "reason": ""}
                                      for index, _ in enumerate(payload["claims"])]}
            else:
                self.send_error(400, "Unknown stage")
                return
            result = {"id": "test", "object": "chat.completion", "created": 1, "model": "mock",
                      "choices": [{"index": 0, "finish_reason": "stop", "message": {
                          "role": "assistant", "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)}}]}
        body = json.dumps(result, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        """Подавляет стандартный журнал HTTP тестовой подмены."""


class PlatformPackageTests(unittest.TestCase):
    """Проверяет воспроизводимость сборки и запуск только упакованных модулей."""

    def test_package_discovers_transitive_modules_and_excludes_answers_and_secrets(self):
        """Сборка следует новым импортам и не переносит данные разработки."""
        modules = discover_modules()
        self.assertIn("rag_pipeline.editions", modules)
        self.assertIn("answer_generation.context", modules)
        self.assertIn("query_processing_module.query_processing.json_utils", modules)
        self.assertNotIn("API.app", modules)
        with tempfile.TemporaryDirectory() as directory:
            first = build_archive(Path(directory) / "first.zip")
            second = build_archive(Path(directory) / "second.zip")
            self.assertEqual(hashlib.sha256(first.read_bytes()).digest(),
                             hashlib.sha256(second.read_bytes()).digest())
            with zipfile.ZipFile(first) as archive:
                names = archive.namelist()
                self.assertFalse(any("/" in name for name in names))
                self.assertIn("_psb_platform_solution_solution.py", names)
                self.assertFalse(any(word in name for name in names
                                     for word in (".env", "demo", "questions.csv", "knowledge-base", "test_")))

    def test_flat_zip_runs_with_shared_platform_env_and_rechecks_saved_result(self):
        """Распакованный ZIP создаёт CSV платформенным движком без исходного проекта."""
        server = ThreadingHTTPServer(("127.0.0.1", 0), PlatformAPI)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, flat = root / "data", root / "flat"
            data.mkdir()
            with zipfile.ZipFile(data / "knowledge-base.zip", "w") as archive:
                archive.writestr("документы компании__Договор поставки П-42-2024.txt",
                                 "ДОГОВОР ПОСТАВКИ № П-42/2024\n1. Платежи\n1.1. Оплата через 10 дней.")
            (data / "questions.csv").write_text("id,question\nq2,Когда оплата?\nq1,Какой срок платежа?\n", encoding="utf-8")
            with zipfile.ZipFile(build_archive(root / "upload.zip")) as archive:
                archive.extractall(flat)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("LLM_", "OPENAI_", "EMB_", "RERANK")) and key != "PYTHONPATH"}
            env.update(DATA_DIR=str(data), ARTIFACTS_DIR=str(root / "artifacts"), MAX_WORKERS="2",
                       LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1", LLM_API_KEY="mock-secret",
                       LLM_MODEL="mock", EMB_MODEL="mock-embedding", RERANK_MODE="none")
            subprocess.run([sys.executable, str(flat / "solution.py")], cwd=flat, env=env,
                           check=True, capture_output=True, text=True, timeout=30)
            with (flat / "submission.csv").open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual([row["id"] for row in rows], ["q2", "q1"])
            self.assertTrue(all("10 дней" in row["answer"] and "п. 1.1" in row["answer"] for row in rows))
            trace = (flat / "answer_trace.jsonl").read_text()
            self.assertNotIn("mock-secret", trace)
            self.assertTrue(all(json.loads(line)["mode"] == "llm" for line in trace.splitlines()))


    def test_flat_zip_recovers_through_all_modular_stages(self):
        """Сбой платформенной генерации запускает поиск и проверку командных модулей."""
        class RecoveryAPI(PlatformAPI):
            """Сохраняет остальные методы прокси при отказе основного промпта."""
            fail_primary = True

        server = ThreadingHTTPServer(("127.0.0.1", 0), RecoveryAPI)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, flat = root / "data", root / "flat"
            data.mkdir()
            with zipfile.ZipFile(data / "knowledge-base.zip", "w") as archive:
                archive.writestr("документы компании__Договор поставки П-42-2024.txt",
                                 "ДОГОВОР ПОСТАВКИ № П-42/2024\n1. Платежи\n1.1. Оплата через 10 дней.")
                archive.writestr("документы компании__Правила.txt", "ПРАВИЛА\n1. Хранение архивных документов.")
            (data / "questions.csv").write_text("id,question\nq1,Когда оплата?\n", encoding="utf-8")
            with zipfile.ZipFile(build_archive(root / "upload.zip")) as archive:
                archive.extractall(flat)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(("LLM_", "OPENAI_", "EMB_", "RERANK")) and key != "PYTHONPATH"}
            env.update(DATA_DIR=str(data), LLM_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                       LLM_API_KEY="mock-secret", LLM_MODEL="mock", USE_EMBEDDINGS="0", MAX_WORKERS="1")
            completed = subprocess.run([sys.executable, str(flat / "solution.py")], cwd=flat, env=env,
                           check=True, capture_output=True, text=True, timeout=30)
            with (flat / "submission.csv").open(encoding="utf-8") as source:
                row = next(csv.DictReader(source))
            trace = json.loads((flat / "answer_trace.jsonl").read_text())
            self.assertIn("10 дней", row["answer"])
            self.assertEqual(trace["mode"], "modular-recovery", completed.stderr)
