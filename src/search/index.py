"""Коллекция фрагментов в Qdrant, своя для каждой модели эмбеддингов."""

import uuid
from dataclasses import dataclass
from qdrant_client import models as qm
from qdrant_client import AsyncQdrantClient

from src.search.ml import TEXT_DIM
from src.core.config import ModelSettings

POINT_NAMESPACE = uuid.UUID("6f1d5d4e-2c1b-4c39-9a8e-0e4f3b7f2a10")


@dataclass(frozen=True, slots=True)
class TextHit:
    section_id: uuid.UUID
    document_id: uuid.UUID
    chunk: str
    content_type: str
    score: float


def collection_name(settings: ModelSettings) -> str:
    return f"chunks__{settings.text_embedder.split('/')[-1].lower()}"


def point_id(document_id: uuid.UUID, section_ordinal: int, chunk_index: int) -> str:
    """Один и тот же фрагмент документа всегда получает один id."""
    return str(uuid.uuid5(POINT_NAMESPACE, f"{document_id}/{section_ordinal}/{chunk_index}"))


async def ensure_collection(client: AsyncQdrantClient, name: str) -> None:
    if await client.collection_exists(name):
        return
    await client.create_collection(name, vectors_config=qm.VectorParams(size=TEXT_DIM, distance=qm.Distance.COSINE))
    # is_tenant: Qdrant размещает точки одной команды рядом и ускоряет поиск
    # с фильтром по команде.
    await client.create_payload_index(
        name, "tenant_id", qm.KeywordIndexParams(type=qm.KeywordIndexType.KEYWORD, is_tenant=True)
    )
    await client.create_payload_index(name, "document_id", qm.PayloadSchemaType.KEYWORD)


def _tenant_condition(tenant_id: uuid.UUID) -> qm.FieldCondition:
    return qm.FieldCondition(key="tenant_id", match=qm.MatchValue(value=str(tenant_id)))


async def delete_document(client: AsyncQdrantClient, name: str, tenant_id: uuid.UUID, document_id: uuid.UUID) -> None:
    document = qm.FieldCondition(key="document_id", match=qm.MatchValue(value=str(document_id)))
    selector = qm.FilterSelector(filter=qm.Filter(must=[_tenant_condition(tenant_id), document]))
    await client.delete(name, points_selector=selector, wait=True)


async def search(
    client: AsyncQdrantClient, name: str, tenant_id: uuid.UUID, vector: list[float], limit: int
) -> list[TextHit]:
    """Поиск фрагментов только среди документов команды."""
    response = await client.query_points(
        name, query=vector, query_filter=qm.Filter(must=[_tenant_condition(tenant_id)]), limit=limit, with_payload=True
    )
    return [
        TextHit(
            section_id=uuid.UUID(point.payload["section_id"]),
            document_id=uuid.UUID(point.payload["document_id"]),
            chunk=point.payload["chunk"],
            content_type=point.payload["content_type"],
            score=point.score
        )
        for point in response.points
    ]
