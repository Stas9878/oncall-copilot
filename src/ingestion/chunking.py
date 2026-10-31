"""Дочерние фрагменты разделов для поиска."""

import re
from datetime import date

from src.ingestion.parsers import ParsedSection

# Меняется при любой правке разбиения: по версии видно, каким разбиением
# проиндексирован документ, и какие документы пора переиндексировать.
CHUNKER_VERSION = "sections-v3"
MAX_CHUNK_CHARS = 450

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[А-ЯA-Z0-9«])")


def _pieces(text: str) -> list[str]:
    """Абзацы и пункты списков. Слишком длинный абзац делится по предложениям."""
    pieces: list[str] = []
    for paragraph in re.split(r"\n\s*\n|\n(?=[•\-*]|\d+\.\s|\|)", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= MAX_CHUNK_CHARS:
            pieces.append(paragraph)
        else:
            pieces.extend(_SENTENCE_END.split(paragraph))
    return pieces


def _pack(pieces: list[str]) -> list[str]:
    """Склеивает соседние части, пока фрагмент не превысит лимит."""
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > MAX_CHUNK_CHARS:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def child_chunks(section: ParsedSection) -> list[str]:
    """Тексты фрагментов раздела. Строка таблицы — отдельный фрагмент."""
    if section.rows:
        return section.rows
    return _pack(_pieces(section.text))


def embedding_text(title: str, heading: str | None, chunk: str) -> str:
    """Текст для эмбеддинга с названием документа и раздела."""
    context = title if not heading or heading == title else f"{title}. {heading}"
    return f"{context}. {chunk}"


_ISO_DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря"
)


def with_spoken_date(name: str) -> str:
    """Дописывает к имени файла дату словами: «grafana_2026-10-12.png, 12 октября 2026»."""
    match = _ISO_DATE.search(name)
    if match is None:
        return name
    try:
        day = date(int(match[1]), int(match[2]), int(match[3]))
    except ValueError:
        return name
    return f"{name}, {day.day} {_MONTHS[day.month - 1]} {day.year}"
