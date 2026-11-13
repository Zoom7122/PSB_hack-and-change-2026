"""Расширение найденных положений соседями, редакциями и связанными документами."""

import re

from data_processing import textproc
from data_processing.contract_adapter import normalize_number
from .editions import applicability, edition_weight


def document_anchors(corpus, question: str) -> set[str]:
    """Привязывает явно указанный номер или полное имя контрагента к документу."""
    tokens = set(textproc.tokenize(question))
    normalized = normalize_number(question) or ""
    matches = set()
    for meta in corpus.metas.values():
        number = normalize_number(meta.number)
        if number and re.search(r"[-/]", number) and re.search(re.escape(number) + r"(?!\d)", normalized):
            matches.add(meta.doc_id)
        party = set(textproc.tokenize(meta.counterparty)) - {"ооо"}
        if len(party) >= 2 and party <= tokens:
            matches.add(meta.doc_id)
    return matches


def _related(corpus, seeds: list[dict]) -> list[tuple[str, str]]:
    """Находит допсоглашения, ту же норму другой редакции и специальные письма."""
    result = []
    for seed in seeds:
        meta = corpus.metas[seed["document_id"]]
        for other in corpus.metas.values():
            parent = normalize_number(other.parent_contract)
            if parent and parent == normalize_number(meta.number):
                result.extend((cid, "related_amendment") for cid, source in corpus.sources.items()
                              if source["document_id"] == other.doc_id)
            if meta.kind == "регламент" and other.kind == "регламент":
                family = seed["document_title"].split(" (ред.")[0]
                if other.doc_id != meta.doc_id and other.citation_title.split(" (ред.")[0] == family:
                    result.extend((cid, "alternate_edition") for cid, source in corpus.sources.items()
                                  if source["document_id"] == other.doc_id and source["section"] == seed["section"])
        if meta.kind == "регламент":
            result.extend((cid, "related_special_rule") for cid, source in corpus.sources.items()
                          if source["document_type"] in {"письмо", "записка"}
                          and "регламент" in source["text"].lower()
                          and seed["section"] in source["text"])
    return list(dict.fromkeys(result))


def expand_context(corpus, seeds: list[dict], question: str, period, *,
                   max_sources: int, max_chars: int) -> list[dict]:
    """Собирает ограниченный по размеру контекст с проверяемыми исходными ID."""
    selected, seen, size = [], set(), 0

    def add(source: dict, reason: str) -> None:
        """Добавляет подлинный фрагмент, если он помещается в бюджет контекста."""
        nonlocal size
        cost = len(source.get("body", source["text"])) + len(source["document_title"])
        if source["chunk_id"] in seen or len(selected) >= max_sources or size + cost > max_chars:
            return
        item = dict(source)
        item["selection_reason"] = reason
        item["applicability"] = applicability(source, period)
        selected.append(item)
        seen.add(item["chunk_id"])
        size += cost

    for seed in seeds:
        add(seed, "retrieved")
    for chunk_id, reason in _related(corpus, seeds):
        add(corpus.sources[chunk_id], reason)
    anchors = document_anchors(corpus, question)
    counts = {}
    for seed in seeds[:4]:
        counts[seed["document_id"]] = counts.get(seed["document_id"], 0) + 1
    focused = anchors | {doc_id for doc_id, count in counts.items() if count >= 2}
    for doc_id in sorted(focused, key=lambda key: (-counts.get(key, 0), key)):
        items = [source for source in corpus.sources.values() if source["document_id"] == doc_id]
        if sum(len(item.get("body", item["text"])) for item in items) <= 10000:
            for source in sorted(items, key=lambda item: -edition_weight(item, period)):
                add(source, "same_document")
    for seed in seeds[:4]:
        chunk = next(item for item in corpus.chunks if item.chunk_id == seed["chunk_id"])
        for neighbor in corpus.chunks:
            if neighbor.doc_id == chunk.doc_id and abs(neighbor.order - chunk.order) == 1:
                add(corpus.sources[neighbor.chunk_id], "adjacent")
    return selected
