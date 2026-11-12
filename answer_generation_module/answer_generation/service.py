"""Модуль 5: Answer Generation + Verification.

Конвейер одного запроса
=======================
  1. подготовка доказательств (evidence.py): дедуп, corpus_version, вычистка
     инъекций, применимость редакций на дату события, сверка чисел вопроса;
  2. гейт: нет пригодных фрагментов или сигналы поиска ниже порогов → отказ
     без вызова LLM (дёшево и честно);
  3. генерация черновика (LLM, JSON по предложениям с источниками);
  4. проверка (verification.py): ссылки и числа — программно, смысл — LLM;
  5. если есть замечания — одно исправление с обратной связью и повторная проверка;
  6. сборка: в ответ попадают только подтверждённые утверждения; если их нет —
     status="insufficient" с объяснением, чего не хватает.

Сервис не бросает исключений из-за модели: при сбое LLM возвращается
корректный GenerationResult со статусом insufficient и пояснением в notes.
Исключение — только невалидный вход (pydantic.ValidationError).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .config import GenerationSettings
from .contracts import (CitationLink, Claim, GenerationRequest, GenerationResult,
                        Verification)
from .evidence import PreparedEvidence, prepare_evidence
from .llm_client import LLMClient
from .numbers import missing_facts
from .parsing import Draft, ParseError, parse_draft, parse_verification
from .prompts import (RETRY_FORMAT, generation_messages, repair_feedback,
                      verification_messages)
from .verification import (Statement, apply_llm_verdicts, apply_rules_only,
                           build_statements, repair_problems, rule_warnings)

logger = logging.getLogger("answer_generation")

TEXT_NO_EVIDENCE = ("В базе знаний не найдено положений, которые относятся к вопросу, "
                    "поэтому достоверно ответить на него нельзя.")
TEXT_NOT_CONFIRMED = ("Найденные положения не позволяют достоверно ответить на вопрос: "
                      "проверка не подтвердила ни одного вывода.")
TEXT_LLM_DOWN = ("Не удалось сформировать проверенный ответ: сервис генерации временно "
                 "недоступен. Повторите запрос позже.")
TEXT_PARTIAL_DROPPED = ("Часть выводов не подтверждена найденными положениями и "
                        "исключена из ответа.")


class _LLMUnavailable(Exception):
    pass


@dataclass
class GenerationOutcome:
    result: GenerationResult
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_debug: bool = False) -> dict[str, Any]:
        data = self.result.model_dump(mode="json")
        if include_debug:
            data["debug"] = self.debug
        return data


class AnswerGenerationService:
    def __init__(self, llm_client: LLMClient, settings: GenerationSettings | None = None,
                 *, today: date | None = None) -> None:
        self._llm = llm_client
        self._settings = settings or GenerationSettings()
        self._today = today

    # ------------------------------------------------------------- public API
    async def generate(self, request: dict | GenerationRequest) -> dict[str, Any]:
        """Контрактный вход → контрактный выход (dict, готовый к JSON)."""
        outcome = await self.run(request)
        return outcome.to_dict()

    async def run(self, request: dict | GenerationRequest) -> GenerationOutcome:
        """То же, что generate, но с отладочной трассировкой."""
        req = GenerationRequest.model_validate(request)
        s = self._settings
        started = time.perf_counter()
        trace: dict[str, Any] = {"llm_calls": 0, "stages": []}

        prepared = prepare_evidence(req, max_chunks=s.max_context_chunks,
                                    max_chars=s.max_chunk_chars,
                                    infer_event_date=s.infer_event_date, today=self._today)
        notes = list(prepared.notes)
        notes += [f"фрагмент {cid} не использован: {why}" for cid, why in prepared.excluded
                  if why != "превышен лимит контекста"]
        trace["evidence"] = [
            {"n": e.index, "chunk_id": e.chunk_id, "label": e.label,
             "applicability": e.applicability, "note": e.applicability_note,
             "injection_removed": e.injection_removed}
            for e in prepared.items
        ]
        trace["event_period"] = prepared.period.describe() if prepared.period else None
        trace["premise_hints"] = prepared.premise_hints

        if not prepared.items:
            return self._finish(req, trace, started, self._insufficient(
                req, TEXT_NO_EVIDENCE, notes + ["не найдены применимые подтверждающие положения"]))

        gate = self._evidence_gate(req)
        if gate:
            return self._finish(req, trace, started, self._insufficient(
                req, TEXT_NO_EVIDENCE, notes + [gate]))

        try:
            draft, messages = await self._draft(req, prepared, trace)
        except _LLMUnavailable as exc:
            return self._finish(req, trace, started, self._insufficient(
                req, TEXT_LLM_DOWN, notes + [str(exc)]))

        statements = await self._verify(req, prepared, draft, trace, notes)
        problems = repair_problems(statements, prepared)
        repairs = 0
        while problems and repairs < s.max_repairs:
            repairs += 1
            trace["stages"].append({"repair": repairs, "problems": problems})
            repair_messages = messages + [
                {"role": "assistant", "content": _draft_json(draft)},
                {"role": "user", "content": repair_feedback(problems)},
            ]
            try:
                new_draft = await self._parse_with_retry(repair_messages, trace, "repair")
            except (_LLMUnavailable, ParseError) as exc:
                notes.append(f"исправление черновика не удалось: {exc}")
                break
            draft, messages = new_draft, repair_messages
            statements = await self._verify(req, prepared, draft, trace, notes)
            problems = repair_problems(statements, prepared)

        result = self._assemble(req, prepared, draft, statements, notes)
        return self._finish(req, trace, started, result)

    # ---------------------------------------------------------------- stages
    def _evidence_gate(self, req: GenerationRequest) -> str | None:
        """Отказ до LLM по сигналам поиска — только если они есть у ВСЕХ чанков."""
        s = self._settings
        if not s.evidence_gate:
            return None
        best_bm25, best_vector = 0.0, 0.0
        for chunk in req.chunks:
            scores = chunk.extra_field("scores")
            if not isinstance(scores, dict):
                return None
            try:
                best_bm25 = max(best_bm25, float(scores.get("bm25") or 0.0))
                best_vector = max(best_vector, float(scores.get("vector") or 0.0))
            except (TypeError, ValueError):
                return None
        if best_bm25 < s.gate_bm25_min and best_vector < s.gate_vector_min:
            return (f"слабые сигналы поиска: BM25 {best_bm25:.2f} < {s.gate_bm25_min}, "
                    f"сходство {best_vector:.2f} < {s.gate_vector_min}")
        return None

    async def _call(self, messages: list[dict[str, str]], trace: dict, stage: str) -> str:
        attempts = self._settings.llm_retries + 1
        last_error = "unknown"
        for attempt in range(attempts):
            trace["llm_calls"] += 1
            t0 = time.perf_counter()
            try:
                reply = await self._llm.complete(messages)
                trace["stages"].append({"call": stage, "seconds": round(time.perf_counter() - t0, 2)})
                return reply
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — клиент может быть чужим
                last_error = type(exc).__name__
                logger.warning("LLM call %s failed (attempt %d): %s", stage, attempt + 1, last_error)
                trace["stages"].append({"call": stage, "error": last_error})
        raise _LLMUnavailable(f"модель недоступна на этапе {stage}: {last_error}")

    async def _parse_with_retry(self, messages: list[dict[str, str]], trace: dict,
                                stage: str) -> Draft:
        reply = await self._call(messages, trace, stage)
        try:
            return parse_draft(reply)
        except ParseError:
            retry = messages + [{"role": "assistant", "content": reply[:4000]},
                                {"role": "user", "content": RETRY_FORMAT}]
            return parse_draft(await self._call(retry, trace, f"{stage}-format"))

    async def _draft(self, req: GenerationRequest, prepared: PreparedEvidence,
                     trace: dict) -> tuple[Draft, list[dict[str, str]]]:
        messages = generation_messages(req.question, prepared, self._settings.max_answer_words)
        try:
            draft = await self._parse_with_retry(messages, trace, "generate")
        except ParseError as exc:
            raise _LLMUnavailable(f"ответ модели не соответствует формату: {exc}") from None
        trace["draft"] = {"status": draft.status,
                          "sentences": [s.__dict__ for s in draft.sentences],
                          "missing_information": draft.missing_information}
        return draft, messages

    async def _verify(self, req: GenerationRequest, prepared: PreparedEvidence, draft: Draft,
                      trace: dict, notes: list[str]) -> list[Statement]:
        statements = build_statements(draft, prepared)
        if not statements:
            return statements
        warnings = rule_warnings(statements, prepared, req.question)
        verdicts = None
        if self._settings.verification_mode == "llm":
            items = [st.as_prompt_item(warnings.get(st.sid, [])) for st in statements]
            try:
                reply = await self._call(verification_messages(req.question, prepared, items),
                                         trace, "verify")
                verdicts = parse_verification(reply)
            except (_LLMUnavailable, ParseError) as exc:
                notes.append(f"LLM-проверка недоступна, применены только правила ({exc})")
        if verdicts is None:
            apply_rules_only(statements, prepared, req.question)
        else:
            apply_llm_verdicts(statements, verdicts, prepared, req.question)
        trace["stages"].append({"verification": [
            {"id": st.sid, "kind": st.kind, "verdict": st.verdict, "sources": st.sources,
             "reason": st.reason, "warnings": warnings.get(st.sid, [])}
            for st in statements]})
        return statements

    # -------------------------------------------------------------- assembly
    def _assemble(self, req: GenerationRequest, prepared: PreparedEvidence, draft: Draft,
                  statements: list[Statement], notes: list[str]) -> GenerationResult:
        index = prepared.by_index()
        claims_kept = [st for st in statements if st.is_claim and st.verdict == "supported"]
        dropped = [st for st in statements if st.verdict in {"unsupported", "gap_wrong"}]
        for st in dropped:
            notes.append(f"исключено из ответа: «{_short(st.text)}» — {st.reason or 'не подтверждено'}")
        invalid_refs = sorted({n for st in statements for n in st.invalid_sources})
        if invalid_refs:
            notes.append(f"модель ссылалась на несуществующие фрагменты: {invalid_refs}")

        kept = [st for st in statements
                if (st.is_claim and st.verdict == "supported") or (st.kind == "gap" and st.verdict == "gap_ok")]

        if not claims_kept:
            unsupported = [st for st in statements if st.is_claim and st.verdict == "unsupported"]
            gaps = [st.text for st in kept if st.kind == "gap"]
            text = " ".join(_sentence(g) for g in gaps) if gaps else None
            if text is None:
                missing = draft.missing_information
                if missing and missing_facts(missing, _all_text(prepared) + req.question):
                    missing = None
                base = TEXT_NOT_CONFIRMED if unsupported else TEXT_NO_EVIDENCE
                text = base + (f" Не хватает сведений: {_lower_first(_sentence(missing))}" if missing else "")
            if unsupported:
                claims = [Claim(claim_id=f"claim-{i}", text=st.text, chunk_ids=[], support="unsupported")
                          for i, st in enumerate(unsupported, start=1)]
                verification = Verification(
                    status="failed",
                    invalid_chunk_ids=[f"#{n}" for n in invalid_refs],
                    unsupported_claim_ids=[c.claim_id for c in claims],
                    notes=notes)
            else:
                claims = []
                verification = Verification(status="not_applicable", notes=notes)
            return GenerationResult(request_id=req.request_id, status="insufficient",
                                    answer=_trim_words(text, self._settings.max_answer_words),
                                    claims=claims, citations=[], verification=verification)

        # Если исключён прямой ответ (первое утверждение) или больше половины
        # утверждений — честно говорим, что ответ неполный.
        claims_total = [st for st in statements if st.is_claim]
        first_dropped = bool(claims_total) and claims_total[0].verdict != "supported"
        if first_dropped or len(claims_kept) * 2 < len(claims_total):
            kept.append(Statement(sid="note", text=TEXT_PARTIAL_DROPPED, kind="gap", verdict="gap_ok"))

        kept = _fit_words(kept, self._settings.max_answer_words)
        claims: list[Claim] = []
        links: dict[str, list[str]] = {}
        for st in kept:
            if not st.is_claim:
                continue
            claim_id = f"claim-{len(claims) + 1}"
            chunk_ids = list(dict.fromkeys(index[n].chunk_id for n in st.sources))
            claims.append(Claim(claim_id=claim_id, text=st.text, chunk_ids=chunk_ids,
                                support="supported"))
            for chunk_id in chunk_ids:
                links.setdefault(chunk_id, []).append(claim_id)
        status = "answered" if draft.status == "answered" else "insufficient"
        return GenerationResult(
            request_id=req.request_id,
            status=status,
            answer=_join(kept),
            claims=claims,
            citations=[CitationLink(chunk_id=cid, claim_ids=ids) for cid, ids in links.items()],
            verification=Verification(status="passed", notes=notes),
        )

    def _insufficient(self, req: GenerationRequest, text: str, notes: list[str]) -> GenerationResult:
        return GenerationResult(request_id=req.request_id, status="insufficient", answer=text,
                                claims=[], citations=[],
                                verification=Verification(status="not_applicable", notes=notes))

    def _finish(self, req: GenerationRequest, trace: dict, started: float,
                result: GenerationResult) -> GenerationOutcome:
        trace["seconds"] = round(time.perf_counter() - started, 2)
        trace["status"] = result.status
        logger.info("request %s: %s, %d claims, %d LLM calls, %.2fs", req.request_id,
                    result.status, len(result.claims), trace["llm_calls"], trace["seconds"])
        return GenerationOutcome(result=result, debug=trace)


# ----------------------------------------------------------------- helpers
def _draft_json(draft: Draft) -> str:
    import json
    return json.dumps({
        "status": draft.status,
        "sentences": [{"text": s.text, "kind": s.kind, "sources": s.sources} for s in draft.sentences],
        "missing_information": draft.missing_information,
    }, ensure_ascii=False)


def _all_text(prepared: PreparedEvidence) -> str:
    return "\n".join(item.haystack() for item in prepared.items)


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "!", "?", "…", ":", ";")) else text + "."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text and not text[:2].isupper() else text


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"


RE_STEP = re.compile(r"^(?:\d{1,2}[).]|[-—•])\s")


def _join(statements: list[Statement]) -> str:
    """Предложения в абзац; шаги «1) …» — каждый с новой строки."""
    out, prev_step = "", False
    for st in statements:
        piece = _sentence(st.text)
        is_step = bool(RE_STEP.match(piece))
        if not out:
            out = piece
        elif is_step or prev_step:
            out += "\n" + piece
        else:
            out += " " + piece
        prev_step = is_step
    return out


def _words(text: str) -> int:
    return len(text.split())


def _fit_words(statements: list[Statement], limit: int) -> list[Statement]:
    """Отбрасываем хвостовые предложения, пока ответ не уложится в лимит слов.

    Пояснение о неполноте (sid="note") сохраняем, если оно было: без него
    усечённый ответ выглядел бы полным.
    """
    note = [st for st in statements if st.sid == "note"]
    body = [st for st in statements if st.sid != "note"]
    # Ответ должен начинаться с подтверждённого утверждения, иначе обрезка может
    # оставить одни «пробелы» без единого вывода.
    first_claim = next((i for i, st in enumerate(body) if st.is_claim), 0)
    if first_claim:
        body = [body[first_claim]] + body[:first_claim] + body[first_claim + 1:]
    budget = limit - sum(_words(st.text) for st in note)
    kept: list[Statement] = []
    used = 0
    for st in body:
        size = _words(st.text)
        if used + size > budget and kept:
            break
        kept.append(st)
        used += size
    if kept and used > budget:            # одно гигантское предложение
        words = kept[0].text.split()[:max(budget, 10)]
        kept[0] = Statement(**{**kept[0].__dict__, "text": " ".join(words).rstrip(",;:") + "…"})
    return kept + note


def _trim_words(text: str, limit: int) -> str:
    words = text.split()
    return text if len(words) <= limit else " ".join(words[:limit]).rstrip(",;:") + "…"
