"""Формат ссылки на фрагмент документа: «п. 3.1» или «НК РФ, ст. 164».

Зачем отдельный модуль
======================
Ссылка нужна в двух местах: в JSON-контракте (поле `cite_label`, её покажет
API) и в тексте ответа. Формат должен быть один — иначе юрист увидит в карточке
источника одно, а в ответе другое.

Раньше эта функция жила в `answering.py` (модуль 5). Для модулей 3–4 это лишняя
зависимость: поиск не должен тянуть генерацию. Здесь только разметка, ноль
сетевых вызовов и ноль импортов из чужих зон.

Правило формата взято из примеров ответов организаторов (`demo_output.csv`):
  * договоры и внутренние документы — номер пункта: «(п. 3.1)»;
  * кодексы и законы — краткое имя и статья: «(ТК РФ, ст. 70)», «(152-ФЗ, ст. 9)»;
  * служебные маркеры нарезки («фрагмент 3», «ч. 1 (продолжение 48)») — не
    ссылки, вместо них название документа.
"""

from __future__ import annotations

import re

RE_LAW_TITLE = re.compile(r"^(ТК РФ|НК РФ|КоАП РФ|ГК РФ|УК РФ|\d+-ФЗ)\s*,?\s*(.*)$")

# Внутренние маркеры нарезки, которые нельзя показывать как ссылку
SERVICE_MARKERS = ("фрагмент", "продолжение", "без номера")


def is_service_marker(section: str) -> bool:
    """Служебный ли это маркер нарезки, а не юридическая ссылка."""
    low = (section or "").strip().lower()
    if not low:
        return True
    return any(marker in low for marker in SERVICE_MARKERS)


def inline_reference(doc_title: str, section: str) -> str:
    """Короткая ссылка для вставки в текст и в `cite_label`.

    >>> inline_reference("Договор аренды № А-08/2024 Северный Склад", "п. 3.1")
    'п. 3.1'
    >>> inline_reference("НК РФ, ст. 164", "ч. 1 (продолжение 48)")
    'НК РФ, ст. 164'
    """
    title = (doc_title or "").strip()
    section = (section or "").strip()
    match = RE_LAW_TITLE.match(title)
    if match:
        code, rest = match.group(1), match.group(2).strip()
        return f"{code}, {rest}" if rest else code
    if is_service_marker(section):
        return title
    return section or title
