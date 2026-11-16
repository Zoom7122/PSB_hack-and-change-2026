"""Веб-чат: API, история, источники и просмотр документов (HTTP к LLM подменяется)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
import httpx  # noqa: E402

from platform_solution import solution as engine  # noqa: E402


def fake_vector(text: str, dim: int = 64) -> list[float]:
    vector = [0.0] * dim
    for token in engine.tokens(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1.0
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / length for x in vector]


class FakeProxy:
    """Имитация OpenAI-совместимого прокси: чат, эмбеддинги, реранкер."""

    def __init__(self, answer) -> None:
        self.answer = answer

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        path = request.url.path
        if path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [{"index": i, "embedding": fake_vector(t)}
                                                      for i, t in enumerate(payload["input"])]})
        if path.endswith("/rerank"):
            q = fake_vector(payload["query"])
            scores = [sum(a * b for a, b in zip(q, fake_vector(d))) for d in payload["documents"]]
            return httpx.Response(200, json={"results": [{"index": i, "relevance_score": v}
                                                         for i, v in enumerate(scores)]})
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={"choices": [{"message": {"content": self.answer(payload["messages"])}}]})
        return httpx.Response(404)


class WebAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["PSB_HISTORY_DB"] = str(Path(cls.tmp.name) / "history.sqlite3")
        import API.app as app_module
        app_module.DB_PATH = Path(os.environ["PSB_HISTORY_DB"])
        cls.app_module = app_module
        from API.engine import LegalAssistant
        cls.LegalAssistant = LegalAssistant

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app_module.set_assistant(None)
        cls.tmp.cleanup()

    def use(self, proxy: FakeProxy | None) -> TestClient:
        settings = engine.Settings(data_dir=ROOT / "data", reranker_model="rerank")
        if proxy is not None:
            settings.base_url, settings.api_key = "http://proxy", "test-key"
        transport = httpx.MockTransport(proxy) if proxy else None
        assistant = self.LegalAssistant(settings, transport, background_index=False)
        self.app_module.set_assistant(assistant)
        client = TestClient(self.app_module.app)
        client.delete("/api/history", headers={"X-Demo-User": "user2"})
        return client

    def test_chat_returns_answer_with_sources_and_saves_history(self) -> None:
        proxy = FakeProxy(lambda m: "АНАЛИЗ: регламент 2025.\nОТВЕТ: Да, нужно согласование (п. 3.1 Регламента).")
        client = self.use(proxy)
        health = client.get("/api/health").json()
        self.assertEqual(health["llm"], "configured")
        self.assertEqual(health["vector_index"], "ready")
        r = client.post("/api/chat", headers={"X-Demo-User": "user2"},
                        json={"question": "Нужно ли согласование финансового директора для договора на 12 млн рублей?",
                              "history": []})
        self.assertEqual(r.status_code, 200, r.text)
        data = r.json()
        self.assertEqual(data["status"], "answered")
        self.assertIn("согласование", data["answer"])
        self.assertTrue(data["sources"])
        titles = " ".join(s["document_title"] for s in data["sources"])
        self.assertIn("Регламент", titles)
        self.assertTrue(any(s["outdated"] for s in data["sources"]), "редакция 2023 должна быть помечена")
        history = client.get("/api/history", headers={"X-Demo-User": "user2"}).json()["messages"]
        self.assertEqual([m["role"] for m in history], ["user", "assistant"])
        self.assertEqual(len(history[1]["sources"]), len(data["sources"]))
        doc = client.get(f"/api/documents/{data['sources'][0]['document_id']}").json()
        self.assertIn("text", doc)
        self.assertEqual(client.get("/api/documents/999").status_code, 404)

    def test_dialog_history_reaches_model_and_injection_is_removed(self) -> None:
        seen = []
        proxy = FakeProxy(lambda m: seen.append(m[-1]["content"]) or "ОТВЕТ: Срок — 3 рабочих дня (п. 3.2).")
        client = self.use(proxy)
        r = client.post("/api/chat", json={"question": "А какая неустойка?", "history": [
            {"role": "user", "content": "Какой срок оплаты по договору П-17/2024?"},
            {"role": "assistant", "content": "Срок оплаты указан в п. 4.1."}]})
        self.assertEqual(r.status_code, 200, r.text)
        prompt = seen[-1]
        self.assertIn("Пользователь: А какая неустойка?", prompt)
        self.assertIn("Ассистент: Срок оплаты указан в п. 4.1.", prompt)
        self.assertIn("Альфа-Трейд", prompt)
        self.assertNotIn("90 календарных дней", prompt)

    def test_without_llm_shows_found_provisions(self) -> None:
        client = self.use(None)
        self.assertEqual(client.get("/api/health").json()["llm"], "not_configured")
        r = client.post("/api/chat", json={"question": "Какой испытательный срок при срочном трудовом договоре?",
                                           "history": []})
        data = r.json()
        self.assertEqual(data["status"], "insufficient")
        self.assertTrue(data["sources"])
        self.assertIn("ТК РФ", " ".join(str(s["section"]) for s in data["sources"]))

    def test_validation(self) -> None:
        client = self.use(None)
        self.assertEqual(client.post("/api/chat", json={"question": "   "}).status_code, 422)
        self.assertEqual(client.post("/api/chat", json={"question": "x", "extra": 1}).status_code, 422)
        self.assertEqual(client.get("/api/history", headers={"X-Demo-User": "hacker"}).status_code, 400)
        self.assertEqual(client.get("/").status_code, 200)

    def test_conversations_are_persistent_and_isolated_by_profile(self) -> None:
        """Диалоги хранят независимую историю и недоступны другим профилям."""
        seen = []
        proxy = FakeProxy(lambda m: seen.append(m[-1]["content"]) or "ОТВЕТ: Условия указаны в договоре.")
        client = self.use(proxy)
        headers = {"X-Demo-User": "user2"}
        first = client.post("/api/conversations", headers=headers).json()["id"]
        second = client.post("/api/conversations", headers=headers).json()["id"]
        for conversation_id, question in [(first, "Вопрос про первый договор"), (second, "Вопрос про второй договор")]:
            response = client.post("/api/chat", headers=headers, json={"question": question, "conversation_id": conversation_id})
            self.assertEqual(response.status_code, 200, response.text)
        response = client.post("/api/chat", headers=headers, json={"question": "А какой срок?", "conversation_id": first})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Вопрос про первый договор", seen[-1])
        self.assertNotIn("Вопрос про второй договор", seen[-1])
        rows = client.get(f"/api/conversations/{first}/messages", headers=headers).json()["messages"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(client.get(f"/api/conversations/{second}/messages", headers=headers).json()["messages"][0]["content"], "Вопрос про второй договор")
        self.assertEqual(client.get(f"/api/conversations/{first}/messages", headers={"X-Demo-User": "user1"}).status_code, 404)
        listing = client.get("/api/conversations", headers=headers).json()["conversations"]
        self.assertTrue(any(item["id"] == first and item["title"] == "Вопрос про первый договор" for item in listing))

    def test_migrates_legacy_history(self) -> None:
        """Старая история становится диалогом без удаления сообщений."""
        path = Path(self.tmp.name) / "legacy.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, user_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT NOT NULL)")
            db.execute("INSERT INTO messages VALUES (1, 'user3', 'user', 'Старый вопрос', '2026-01-01')")
        original = self.app_module.DB_PATH
        self.app_module.DB_PATH = path
        try:
            client = TestClient(self.app_module.app)
            headers = {"X-Demo-User": "user3"}
            conversations = client.get("/api/conversations", headers=headers).json()["conversations"]
            self.assertEqual(len(conversations), 1)
            result = client.get(f"/api/conversations/{conversations[0]['id']}/messages", headers=headers).json()
            self.assertEqual(result["messages"][0]["content"], "Старый вопрос")
            self.assertEqual(len(client.get("/api/conversations", headers=headers).json()["conversations"]), 1)
        finally:
            self.app_module.DB_PATH = original

    def test_rename_is_isolated_and_survives_first_answer(self) -> None:
        """Новое название не сбрасывается генерацией заголовка первого вопроса."""
        client = self.use(None)
        headers = {"X-Demo-User": "user2"}
        conversation_id = client.post("/api/conversations", headers=headers).json()["id"]
        path = f"/api/conversations/{conversation_id}"
        self.assertEqual(client.patch(path, headers=headers, json={"title": " Новый  диалог "}).status_code, 200)
        self.assertEqual(client.patch(path, headers={"X-Demo-User": "user1"}, json={"title": "Чужой"}).status_code, 404)
        self.assertEqual(client.patch(path, headers=headers, json={"title": "  "}).status_code, 422)
        self.assertEqual(client.post("/api/chat", headers=headers, json={"question": "Первый вопрос", "conversation_id": conversation_id}).status_code, 200)
        listing = client.get("/api/conversations", headers=headers).json()["conversations"]
        self.assertEqual(next(row["title"] for row in listing if row["id"] == conversation_id), "Новый диалог")

    def test_two_conversations_can_generate_simultaneously(self) -> None:
        """Независимые запросы выполняются одновременно и сохраняются по своим диалогам."""
        client = self.use(None)
        headers = {"X-Demo-User": "user2"}
        first = client.post("/api/conversations", headers=headers).json()["id"]
        second = client.post("/api/conversations", headers=headers).json()["id"]
        both_started = threading.Barrier(2, timeout=5)

        class SlowAssistant:
            """Имитирует модель, которая одновременно ждёт два запроса."""

            def ask(self, question, history):
                """Дожидается второго запроса и возвращает ответ с текстом вопроса."""
                both_started.wait()
                return {"status": "answered", "answer": "Ответ: " + question, "sources": []}

        self.app_module.set_assistant(SlowAssistant())
        def send(conversation_id, question):
            """Отправляет вопрос в один из созданных диалогов."""
            return client.post("/api/chat", headers=headers, json={"question": question, "conversation_id": conversation_id})
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_response = pool.submit(send, first, "Первый")
            second_response = pool.submit(send, second, "Второй")
            self.assertEqual(first_response.result(timeout=10).status_code, 200)
            self.assertEqual(second_response.result(timeout=10).status_code, 200)
        for conversation_id, question in [(first, "Первый"), (second, "Второй")]:
            rows = client.get(f"/api/conversations/{conversation_id}/messages", headers=headers).json()["messages"]
            self.assertEqual([row["content"] for row in rows], [question, "Ответ: " + question])


if __name__ == "__main__":
    unittest.main()
