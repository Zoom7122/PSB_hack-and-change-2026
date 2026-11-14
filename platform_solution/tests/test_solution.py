"""Офлайн-тесты платформенного решения: HTTP подменяется httpx.MockTransport."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import solution as s  # noqa: E402

KB_ZIP = ROOT.parent / "data" / "knowledge-base.zip"
QUESTIONS = HERE / "demo_questions.csv"


def fake_vector(text: str, dim: int = 64) -> list[float]:
    """Детерминированный «эмбеддинг»: мешок хэшированных основ слов."""
    vector = [0.0] * dim
    for token in s.tokens(text):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % dim] += 1.0
    length = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / length for x in vector]


class FakeProxy:
    """Имитация OpenAI-совместимого прокси платформы."""

    def __init__(self, answer=None, fail_chat=False, fail_embeddings=False, fail_rerank=False):
        self.answer = answer or (lambda messages: "АНАЛИЗ: документ найден.\nОТВЕТ: Да. Срок — 3 рабочих дня (п. 3.2).")
        self.fail_chat, self.fail_embeddings, self.fail_rerank = fail_chat, fail_embeddings, fail_rerank
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        path = request.url.path
        self.calls.append((path, payload))
        assert request.headers["Authorization"] == "Bearer test-key"
        if path.endswith("/embeddings"):
            if self.fail_embeddings:
                return httpx.Response(500, json={"error": "down"})
            data = [{"index": i, "embedding": fake_vector(t)} for i, t in enumerate(payload["input"])]
            return httpx.Response(200, json={"data": data})
        if path.endswith("/rerank"):
            if self.fail_rerank:
                return httpx.Response(404, json={"error": "no model"})
            q = fake_vector(payload["query"])
            scores = [sum(a * b for a, b in zip(q, fake_vector(d))) for d in payload["documents"]]
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            return httpx.Response(200, json={"results": [{"index": i, "relevance_score": scores[i]} for i in order]})
        if path.endswith("/chat/completions"):
            if self.fail_chat:
                return httpx.Response(503, json={"error": "busy"})
            content = self.answer(payload["messages"])
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Response(404)


class RunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "data").mkdir()
        shutil.copy(KB_ZIP, self.tmp / "data" / "knowledge-base.zip")
        shutil.copy(QUESTIONS, self.tmp / "data" / "questions.csv")
        s.Api.post.__defaults__  # noqa: B018 - модуль загружен
        self._sleep = s.time.sleep
        s.time.sleep = lambda _: None          # без пауз между повторами

    def tearDown(self) -> None:
        s.time.sleep = self._sleep
        shutil.rmtree(self.tmp, ignore_errors=True)

    def settings(self, **kwargs) -> s.Settings:
        base = dict(data_dir=self.tmp / "data", base_url="http://proxy/v1", api_key="test-key",
                    reranker_model="qwen3-reranker-8b", workers=3, output=self.tmp / "submission.csv",
                    trace=self.tmp / "trace.jsonl")
        base.update(kwargs)
        return s.Settings(**base)

    def read_submission(self) -> list[dict]:
        with (self.tmp / "submission.csv").open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ["id", "answer"])
            return list(reader)

    def test_full_run_writes_valid_submission(self) -> None:
        proxy = FakeProxy()
        outcomes = s.run(self.settings(), httpx.MockTransport(proxy), today=date(2026, 11, 13))
        rows = self.read_submission()
        self.assertEqual([r["id"] for r in rows], [str(i) for i in range(1, 19)])
        self.assertTrue(all(r["answer"].strip() for r in rows))
        self.assertTrue(all(o.mode in ("llm", "llm-repaired") for o in outcomes))
        self.assertEqual(rows[0]["answer"], "Да. Срок — 3 рабочих дня (п. 3.2).")
        paths = [p for p, _ in proxy.calls]
        self.assertTrue(any(p.endswith("/embeddings") for p in paths))
        self.assertTrue(any(p.endswith("/rerank") for p in paths))
        self.assertTrue((self.tmp / "trace.jsonl").exists())

    def test_prompt_contains_sanitized_documents_and_question(self) -> None:
        seen = []

        def answer(messages):
            seen.append(messages[1]["content"])
            return "АНАЛИЗ: …\nОТВЕТ: Нет."
        s.run(self.settings(), httpx.MockTransport(FakeProxy(answer)), today=date(2026, 11, 13))
        joined = "\n".join(seen)
        self.assertNotIn("ИИ-АССИСТЕНТА", joined)          # строка-инъекция вырезана
        self.assertNotIn("90 календарных дней", joined)
        self.assertIn("Текущая дата: 13.11.2026", joined)
        self.assertIn("УТРАТИЛ СИЛУ с 01.03.2025", joined)

    def test_llm_down_falls_back_to_extractive_answers(self) -> None:
        outcomes = s.run(self.settings(), httpx.MockTransport(FakeProxy(fail_chat=True)))
        rows = self.read_submission()
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r["answer"].strip() for r in rows))
        self.assertTrue(all(o.mode.startswith("extractive") for o in outcomes))

    def test_embeddings_and_reranker_failures_are_not_fatal(self) -> None:
        proxy = FakeProxy(fail_embeddings=True, fail_rerank=True)
        outcomes = s.run(self.settings(), httpx.MockTransport(proxy))
        self.assertTrue(all(o.mode == "llm" for o in outcomes))
        rerank_calls = [p for p, _ in proxy.calls if p.endswith("/rerank")]
        self.assertLessEqual(len(rerank_calls), 3 * 2)       # после 3 сбоев реранкер выключается

    def test_exhausted_time_budget_skips_llm_but_answers_everything(self) -> None:
        proxy = FakeProxy()
        settings = self.settings(time_budget=0)
        outcomes = s.run(settings, httpx.MockTransport(proxy))
        self.assertEqual(len(self.read_submission()), 18)
        self.assertTrue(all(o.mode == "extractive-deadline" for o in outcomes))
        self.assertFalse(any(p.endswith("/chat/completions") for p, _ in proxy.calls))

    def test_empty_content_retries_with_larger_token_limit(self) -> None:
        budgets = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            budgets.append(payload["max_tokens"])
            content = "" if len(budgets) == 1 else "ОТВЕТ: Да."
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        api = s.Api(self.settings(), httpx.MockTransport(handler))
        self.assertEqual(api.chat([{"role": "user", "content": "x"}]), "ОТВЕТ: Да.")
        self.assertGreater(budgets[1], budgets[0])

    def test_no_api_still_writes_file(self) -> None:
        s.run(self.settings(base_url="", api_key=""))
        self.assertEqual(len(self.read_submission()), 18)

    def test_fabricated_number_triggers_one_repair(self) -> None:
        calls = []

        def answer(messages):
            calls.append(messages)
            if len(messages) == 2:
                return "АНАЛИЗ: …\nОТВЕТ: Оплата в течение 90 календарных дней."
            return "АНАЛИЗ: исправлено\nОТВЕТ: Оплата в течение 30 календарных дней с даты подписания УПД (п. 4.2)."
        q = s.Question.parse("1", "Пользователь: П-17/2024. В какой срок оплатить товар?")
        kb = s.KnowledgeBase.from_sources(s.read_sources(KB_ZIP))
        api = s.Api(self.settings(), httpx.MockTransport(FakeProxy(answer)))
        assistant = s.Assistant(kb, s.Retriever(kb, api, self.settings()), api, self.settings(),
                                deadline=s.time.monotonic() + 600)
        outcome = assistant.answer(q)
        self.assertEqual(outcome.mode, "llm-repaired")
        self.assertIn("30 календарных дней", outcome.answer)
        self.assertEqual(len(calls), 2)
        self.assertIn("90", calls[1][-1]["content"])

    def assistant(self, answer) -> s.Assistant:
        kb = s.KnowledgeBase.from_sources(s.read_sources(KB_ZIP))
        api = s.Api(self.settings(), httpx.MockTransport(FakeProxy(answer)))
        return s.Assistant(kb, s.Retriever(kb, api, self.settings()), api, self.settings(),
                           deadline=s.time.monotonic() + 600)

    def test_missing_clause_reference_triggers_repair(self) -> None:
        calls = []

        def answer(messages):
            calls.append(messages)
            if len(messages) == 2:
                return "АНАЛИЗ: …\nОТВЕТ: Оплата в течение 30 календарных дней (п. 9.7)."
            return "АНАЛИЗ: исправлено\nОТВЕТ: Оплата в течение 30 календарных дней с даты подписания УПД (п. 4.2)."
        outcome = self.assistant(answer).answer(s.Question.parse("1", "П-17/2024. В какой срок оплатить товар?"))
        self.assertEqual(outcome.mode, "llm-repaired")
        self.assertIn("(п. 4.2)", outcome.answer)
        self.assertIn("п. 9.7", calls[1][-1]["content"])

    def test_repair_that_does_not_help_is_discarded(self) -> None:
        def answer(messages):
            if len(messages) == 2:
                return "ОТВЕТ: Оплата в течение 30 календарных дней (п. 9.7)."
            return "ОТВЕТ: Оплата через 30 дней, см. п. 9.8."
        outcome = self.assistant(answer).answer(s.Question.parse("1", "П-17/2024. В какой срок оплатить товар?"))
        self.assertEqual(outcome.mode, "llm")
        self.assertIn("п. 9.7", outcome.answer)

    def test_analysis_without_answer_part_is_completed(self) -> None:
        calls = []

        def answer(messages):
            calls.append(messages)
            if len(messages) == 2:
                return "АНАЛИЗ: договор П-17/2024, пункт 4.2, предпосылка верна"
            return "ОТВЕТ: Оплата в течение 30 календарных дней с даты подписания УПД (п. 4.2)."
        outcome = self.assistant(answer).answer(s.Question.parse("1", "П-17/2024. В какой срок оплатить товар?"))
        self.assertEqual(outcome.answer, "Оплата в течение 30 календарных дней с даты подписания УПД (п. 4.2).")
        self.assertIn("no-answer-part", outcome.issues)
        self.assertEqual(len(calls), 2)

    def test_truncated_answer_is_retried_with_larger_token_limit(self) -> None:
        budgets = []

        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content)
            budgets.append(payload["max_tokens"])
            if len(budgets) == 1:
                return httpx.Response(200, json={"choices": [{"message": {"content": "АНАЛИЗ: начал"},
                                                              "finish_reason": "length"}]})
            return httpx.Response(200, json={"choices": [{"message": {"content": "ОТВЕТ: Да."},
                                                          "finish_reason": "stop"}]})
        api = s.Api(self.settings(), httpx.MockTransport(handler))
        self.assertEqual(api.chat([{"role": "user", "content": "x"}]), "ОТВЕТ: Да.")
        self.assertGreater(budgets[1], budgets[0])


class KnowledgeBaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kb = s.KnowledgeBase.from_sources(s.read_sources(KB_ZIP))
        cls.retriever = s.Retriever(cls.kb, None, s.Settings())

    def doc(self, fragment: str) -> s.Document:
        return next(d for d in self.kb.documents if fragment in d.name)

    def context_docs(self, text: str) -> list[str]:
        ctx = s.build_context(self.kb, self.retriever.search(s.Question.parse("x", text)))
        return [self.kb.documents[i].name for i in ctx.doc_ids]

    def test_metadata(self) -> None:
        self.assertIn("УТРАТИЛ СИЛУ с 01.03.2025", self.doc("ред 2023").status)
        self.assertIn("действует с 01.03.2025", self.doc("ред 2025").status)
        self.assertEqual(self.doc("Доп соглашение").parent_numbers, {"п17/2024"})
        self.assertEqual(self.doc("ТК ст70").kind, "law")
        self.assertEqual(s._law_citation(self.doc("ТК ст70").title), "ТК РФ, ст. 70")
        self.assertEqual(s._law_citation(self.doc("152ФЗ").title), "ст. 9 Федерального закона № 152-ФЗ")
        self.assertTrue(self.doc("Служебная записка").removed_lines)

    def test_similar_counterparties_are_not_mixed(self) -> None:
        trading = self.context_docs("По договору с «Альфа-Трейдинг» можем оплатить авансом?")
        self.assertTrue(trading[0].startswith("Договор поставки ПТ-05-2023"))
        self.assertFalse(any("П-17-2024" in n for n in trading))
        trade = self.context_docs("«Альфа-Трейд» привёз мебель с дефектами. Что приложить к претензии?")
        self.assertTrue(trade[0].startswith("Договор поставки П-17-2024"))
        self.assertIn("Доп соглашение 1 к договору П-17-2024", trade)
        self.assertFalse(any("ПТ-05" in n for n in trade))

    def test_abbreviation_finds_law_written_in_full(self) -> None:
        docs = self.context_docs("Какой МРОТ установлен?")
        self.assertTrue(docs[0].startswith("МРОТ"), docs)

    def test_contract_question_does_not_pull_unrelated_laws(self) -> None:
        docs = self.context_docs("П-17/2024. Доставка и упаковка уже включены в цену?")
        self.assertFalse(any(self.doc(n).kind == "law" for n in docs), docs)

    def test_citation_check(self) -> None:
        context = "3.1. Арендная плата составляет 450 000 рублей.\nСтатья 70. Испытание при приеме на работу\n" \
                  "в случаях, указанных в статьях 6, 10 и 11 настоящего закона"
        ok = "Плата — 450 000 рублей (п. 3.1). Источник: ТК РФ, ст. 70. Статьи 6, 10 и 11 в базе отсутствуют."
        self.assertEqual(s.ungrounded(ok, context), [])
        self.assertEqual(s.ungrounded_citations("См. п. 9.7 и ст. 614 ГК РФ.", context), ["п. 9.7", "ст. 614"])
        self.assertEqual(s.ungrounded_citations("По договору № Б-99/2021 всё иначе.", context), ["Б-99/2021"])

    def test_inflected_counterparty_name(self) -> None:
        docs = self.context_docs("По договору с «Северным Складом» нас должны пропускать ночью?")
        self.assertTrue(docs[0].startswith("Договор аренды А-08-2024"))

    def test_dialog_uses_last_user_turn_for_counterparty(self) -> None:
        text = ("Пользователь: Подскажите гарантию по договору с «Альфой».\n"
                "Ассистент: Уточните, пожалуйста, поставщик — «Альфа-Трейд» или «Альфа-Трейдинг»?\n"
                "Пользователь: «Альфа-Трейдинг», у них компьютеры покупали. На сколько гарантия?")
        docs = self.context_docs(text)
        self.assertTrue(docs[0].startswith("Договор поставки ПТ-05-2023"))
        self.assertFalse(any("П-17-2024" in n for n in docs))

    def test_regulation_editions_come_together(self) -> None:
        docs = self.context_docs("Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли одобрение генерального директора?")
        self.assertIn("Регламент согласования договоров ред 2023 (утратил силу)", docs)
        self.assertIn("Регламент согласования договоров ред 2025", docs)


class TextTest(unittest.TestCase):
    def test_extract_answer(self) -> None:
        self.assertEqual(s.extract_answer("АНАЛИЗ: x\ny\nОТВЕТ: Да. Можно."), "Да. Можно.")
        self.assertEqual(s.extract_answer("**АНАЛИЗ:** x\n\n**ОТВЕТ:** **Да**, можно."), "Да, можно.")
        self.assertEqual(s.extract_answer("Просто ответ без меток."), "Просто ответ без меток.")
        self.assertEqual(s.extract_answer("<think>скрыто</think>ОТВЕТ: Нет."), "Нет.")

    def test_ungrounded_numbers(self) -> None:
        ctx = "3.1. Арендная плата составляет 450 000 (четыреста пятьдесят тысяч) рублей. Срок — шести месяцев. 15 000 000 рублей"
        self.assertEqual(s.ungrounded_numbers("Плата 450 000 рублей (п. 3.1), срок 6 месяцев, порог 15 млн.", ctx), [])
        self.assertEqual(s.ungrounded_numbers("Плата 500 000 рублей.", ctx), ["500 000"])
        self.assertEqual(s.ungrounded_numbers("Оплата за 90 дней.", ctx), ["90"])

    def test_fit_length(self) -> None:
        text = " ".join(["Слово слово слово."] * 80)
        self.assertLessEqual(s.word_count(s.fit_length(text)), 150)

    def test_doc_numbers(self) -> None:
        self.assertEqual(s.doc_numbers("договор № П-17/2024 и А-08-2024"), {"п17/2024", "а8/2024"})
        self.assertEqual(s.doc_numbers("ПТ-05/2023"), {"пт5/2023"})

    def test_question_parse(self) -> None:
        q = s.Question.parse("1", "Пользователь: A?\nАссистент: B.\nПользователь: C?")
        self.assertEqual(q.last, "C?")
        self.assertEqual(q.user_turns, ["A?", "C?"])
        self.assertEqual(q.assistant_turns, ["B."])


if __name__ == "__main__":
    unittest.main()
