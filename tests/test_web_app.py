"""Веб-чат: API, история, источники и просмотр документов (HTTP к LLM подменяется)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
