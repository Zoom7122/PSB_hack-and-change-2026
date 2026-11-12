"""Стыковка на настоящих фикстурах соседних модулей (ветка data-processing).

tests/data/module4_rerank_result.json и module2_search_request.json скопированы
без изменений из origin/data-processing:tests/fixtures/.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from answer_generation.adapters import build_generation_request, sources_for_response
from answer_generation.config import GenerationSettings
from answer_generation.contracts import GenerationRequest, GenerationResult
from answer_generation.service import AnswerGenerationService

from tests.fixtures import MockLLM, draft, verdicts

DATA = Path(__file__).parent / "data"


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


class Module4FixtureTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.rerank = load("module4_rerank_result.json")
        self.query = load("module2_search_request.json")

    def test_build_request_from_real_fixtures(self):
        request = build_generation_request(self.rerank, self.query)
        parsed = GenerationRequest.model_validate(request)
        self.assertEqual(parsed.question, self.query["search_request"]["standalone_query"])
        self.assertEqual(parsed.corpus_version, "kb-v1")
        self.assertEqual(len(parsed.chunks), len(self.rerank["chunks"]))
        self.assertEqual(parsed.chunks[0].extra_field("cite_label"), self.rerank["chunks"][0]["cite_label"])

    async def test_full_run_and_sources(self):
        request = build_generation_request(self.rerank, self.query)
        lease = next(i for i, c in enumerate(self.rerank["chunks"], start=1)
                     if c["document_number"] == "А-08/2024" and "6.2" in (c["section"] or ""))
        llm = MockLLM(
            draft(("По п. 6.2 договора № А-08/2024 договор может быть расторгнут по соглашению Сторон, а также "
                   "в иных случаях, предусмотренных законодательством.", "fact", [lease])),
            verdicts(("s1", "supported", [lease], "")),
        )
        outcome = await AnswerGenerationService(llm, GenerationSettings()).run(request)
        result = outcome.to_dict()
        GenerationResult.model_validate(result)
        self.assertEqual(result["status"], "answered")
        sources = sources_for_response(result, self.rerank)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["document_title"], "Договор аренды № А-08/2024 Северный Склад")
        self.assertIsInstance(sources[0]["source_path"], str)       # в модуле 1 это str, не null
        self.assertIn("расторгнут", sources[0]["text"])

    def test_question_required(self):
        with self.assertRaises(ValueError):
            build_generation_request({"chunks": []}, None)


if __name__ == "__main__":
    unittest.main()
