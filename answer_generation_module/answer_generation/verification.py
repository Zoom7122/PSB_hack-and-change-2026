"""Проверка черновика: правила + (опционально) LLM-проверка каждого утверждения.

Уровни проверки
===============
1. Программно — ссылки: номер фрагмента существует среди переданных
   (контракт: «chunk_id существует среди переданных chunks»).
2. Программно — числа: каждое число/дата/номер статьи из утверждения есть в
   тексте фрагментов-источников. Число, которого нет ни в источниках, ни в
   вопросе, — жёсткий отказ утверждению, LLM это не перекрывает.
3. LLM — смысл: подтверждает ли текст фрагмента утверждение, применима ли
   редакция, верно ли утверждение «сведений нет» (защита от ложных отказов).
   Проверяющий может поправить привязку к источникам — так спасаются верные
   утверждения с перепутанным номером фрагмента.

Если LLM-проверка недоступна, решение принимается по правилам 1–2, строже:
утверждение без источников или с «чужими» числами отбрасывается.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .evidence import PreparedEvidence
from .numbers import missing_facts
from .parsing import Draft, VerifierItem


@dataclass
class Statement:
    sid: str
    text: str
    kind: str                                   # fact | conclusion | gap
    sources: list[int] = field(default_factory=list)
    invalid_sources: list[int] = field(default_factory=list)
    verdict: str = "pending"                    # supported | unsupported | gap_ok | gap_wrong
    reason: str = ""
    gap_sources: list[int] = field(default_factory=list)

    @property
    def is_claim(self) -> bool:
        return self.kind in {"fact", "conclusion"}

    def as_prompt_item(self, warnings: list[str]) -> dict:
        item = {"id": self.sid, "kind": self.kind, "text": self.text, "sources": self.sources}
        if warnings:
            item["warnings"] = warnings
        return item


def build_statements(draft: Draft, prepared: PreparedEvidence) -> list[Statement]:
    valid = set(prepared.by_index())
    statements = []
    for number, sentence in enumerate(draft.sentences, start=1):
        good = [n for n in sentence.sources if n in valid]
        bad = [n for n in sentence.sources if n not in valid]
        kind = sentence.kind
        if kind == "gap" and good:
            # «gap» со ссылками — по сути утверждение о содержании документа.
            kind = "conclusion"
        statements.append(Statement(sid=f"s{number}", text=sentence.text, kind=kind,
                                    sources=good, invalid_sources=bad))
    return statements


def _haystack(prepared: PreparedEvidence, sources: list[int]) -> str:
    """Тексты источников плюс других фрагментов ТЕХ ЖЕ документов.

    Реквизиты редакции («ред. от 10.01.2023») обычно лежат в шапке документа,
    а модель ссылается на пункт. Числа из чужих документов не засчитываются —
    это и ловит смешение «Альфа-Трейд» / «Альфа-Трейдинг».
    """
    index = prepared.by_index()
    documents = {index[n].chunk.document_id for n in sources if n in index}
    documents.discard(None)
    picked = [item for item in prepared.items
              if item.index in sources or (item.chunk.document_id in documents)]
    return "\n".join(item.haystack() for item in picked)


def _all_evidence(prepared: PreparedEvidence) -> str:
    return "\n".join(item.haystack() for item in prepared.items)


@dataclass
class NumberCheck:
    hard: list[str]     # нет ни в источниках, ни в вопросе
    soft: list[str]     # нет в источниках, но есть в вопросе


def check_numbers(statement: Statement, prepared: PreparedEvidence, question: str) -> NumberCheck:
    if statement.kind == "gap":
        missing = missing_facts(statement.text, _all_evidence(prepared) + "\n" + question)
        return NumberCheck(hard=missing, soft=[])
    missing = missing_facts(statement.text, _haystack(prepared, statement.sources))
    if not missing:
        return NumberCheck([], [])
    from_question = set(missing_facts(statement.text, question))
    hard = [m for m in missing if m in from_question]      # нет и в вопросе
    soft = [m for m in missing if m not in from_question]  # взято из вопроса
    return NumberCheck(hard=hard, soft=soft)


def rule_warnings(statements: list[Statement], prepared: PreparedEvidence,
                  question: str) -> dict[str, list[str]]:
    warnings: dict[str, list[str]] = {}
    for st in statements:
        items: list[str] = []
        if st.invalid_sources:
            items.append(f"ссылки на несуществующие фрагменты: {st.invalid_sources}")
        if st.is_claim and not st.sources:
            items.append("нет ссылок на фрагменты")
        numbers = check_numbers(st, prepared, question)
        if numbers.hard:
            items.append(f"числа {', '.join(numbers.hard)} не встречаются в источниках")
        if numbers.soft and st.kind == "fact":
            items.append(f"числа {', '.join(numbers.soft)} взяты из вопроса, а не из источников")
        if items:
            warnings[st.sid] = items
    return warnings


def apply_rules_only(statements: list[Statement], prepared: PreparedEvidence, question: str) -> None:
    for st in statements:
        numbers = check_numbers(st, prepared, question)
        if st.kind == "gap":
            st.verdict = "unsupported" if numbers.hard else "gap_ok"
            st.reason = "в утверждении об отсутствии сведений есть числа не из базы" if numbers.hard else ""
            continue
        if not st.sources:
            st.verdict, st.reason = "unsupported", "нет ссылок на существующие фрагменты"
        elif numbers.hard or (numbers.soft and st.kind == "fact"):
            st.verdict = "unsupported"
            st.reason = f"числа {', '.join(numbers.hard + numbers.soft)} отсутствуют в источниках"
        else:
            st.verdict, st.reason = "supported", ""


def apply_llm_verdicts(statements: list[Statement], verdicts: dict[str, VerifierItem],
                       prepared: PreparedEvidence, question: str) -> None:
    valid = set(prepared.by_index())
    pending: list[Statement] = []
    for st in statements:
        item = verdicts.get(st.sid)
        if item is None:
            pending.append(st)
            continue
        if st.kind == "gap":
            numbers = check_numbers(st, prepared, question)
            if numbers.hard:
                st.verdict, st.reason = "unsupported", "в утверждении об отсутствии сведений есть числа не из базы"
            elif item.verdict == "gap_wrong":
                st.verdict, st.reason = "gap_wrong", item.reason or "сведения есть во фрагментах"
                st.gap_sources = [n for n in item.sources if n in valid]
            else:
                st.verdict = "gap_ok"
            continue
        if item.verdict != "supported":
            st.verdict, st.reason = "unsupported", item.reason or "не подтверждается фрагментами"
            continue
        corrected = [n for n in item.sources if n in valid] or st.sources
        if not corrected:
            st.verdict, st.reason = "unsupported", "нет ссылок на существующие фрагменты"
            continue
        st.sources = corrected
        numbers = check_numbers(st, prepared, question)
        if numbers.hard:
            st.verdict = "unsupported"
            st.reason = f"числа {', '.join(numbers.hard)} отсутствуют в источниках"
        else:
            st.verdict, st.reason = "supported", ""
    # Проверяющий пропустил утверждение — решаем по правилам, а не на веру.
    if pending:
        apply_rules_only(pending, prepared, question)


def repair_problems(statements: list[Statement], prepared: PreparedEvidence) -> list[str]:
    problems: list[str] = []
    for st in statements:
        if st.verdict == "unsupported":
            label = "Вывод" if st.kind == "conclusion" else ("Фраза об отсутствии сведений" if st.kind == "gap" else "Утверждение")
            problems.append(f"{label} «{st.text}» не подтверждается: {st.reason}.")
        elif st.verdict == "gap_wrong":
            where = f" (фрагменты {', '.join(map(str, st.gap_sources))})" if st.gap_sources else ""
            problems.append(f"Фраза «{st.text}» неверна: эти сведения есть во фрагментах{where}. "
                            f"Ответь по ним.")
        if st.invalid_sources and st.verdict != "supported":
            problems.append(f"В «{st.text}» ссылки на несуществующие фрагменты {st.invalid_sources}.")
    return problems
