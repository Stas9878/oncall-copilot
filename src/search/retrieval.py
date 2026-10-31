"""Поиск источников для ответа."""

import uuid
from sqlalchemy import select
from dataclasses import dataclass

from src.core.runtime import Resources
from src.security.auth import Principal
from src.search.index import collection_name, search
from src.db.models import Document, DocumentKind, Section

# Реранкер переупорядочивает только кандидатов векторного поиска, поэтому их берётся
# с запасом. 50 пар кросс-энкодер оценивает примерно за 200 мс на CPU.
CANDIDATES = 50


@dataclass(frozen=True, slots=True)
class Source:
    section_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    kind: DocumentKind
    heading: str | None
    page_number: int | None
    content_type: str
    text: str
    score: float


async def _load_sections(
    resources: Resources, tenant_id: uuid.UUID, section_ids: list[uuid.UUID]
) -> dict[uuid.UUID, tuple[Section, Document]]:
    if not section_ids:
        return {}
    async with resources.sessions() as session:
        rows = await session.execute(
            select(Section, Document)
            .join(Document, Document.id == Section.document_id)
            # Команда проверяется ещё раз: если фильтр в Qdrant когда-нибудь
            # сломается, чужой раздел всё равно не попадёт в промпт.
            .where(Section.id.in_(section_ids), Section.tenant_id == tenant_id)
        )
    return {section.id: (section, document) for section, document in rows.tuples()}


async def retrieve(
    resources: Resources, principal: Principal, query: str, *, limit: int, min_score: float
) -> list[Source]:
    """Лучшие разделы команды по запросу. Разделы ниже порога не возвращаются."""
    models = resources.models
    [vector] = await models.embed_texts([query])
    hits = await search(
        resources.qdrant, collection_name(resources.settings.models), principal.tenant_id, vector, CANDIDATES
    )
    scores = await models.rerank(query, [hit.chunk for hit in hits])

    best: dict[uuid.UUID, float] = {}
    for hit, score in zip(hits, scores, strict=True):
        if score >= min_score and score > best.get(hit.section_id, float("-inf")):
            best[hit.section_id] = score
    ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)[:limit]

    loaded = await _load_sections(resources, principal.tenant_id, [section_id for section_id, _ in ranked])
    sources: list[Source] = []
    for section_id, score in ranked:
        if section_id not in loaded:
            continue
        section, document = loaded[section_id]
        sources.append(
            Source(
                section_id=section.id,
                document_id=document.id,
                filename=document.filename,
                kind=document.kind,
                heading=section.heading,
                page_number=section.page_number,
                content_type=section.content_type,
                text=section.text,
                score=round(score, 3)
            )
        )
    return sources
