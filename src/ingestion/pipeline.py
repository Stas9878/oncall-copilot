"""Индексация одного документа: разбор, фрагменты, векторы, запись."""

import uuid
from sqlalchemy import delete
from qdrant_client import models as qm

from src.core.logs import log
from src.core.runtime import Resources
from src.core.storage import FileStorage
from src.generation.cache import bump_kb_version
from src.ingestion.captions import describe_image, image_mime
from src.db.models import Document, DocumentKind, DocumentStatus, Section
from src.search.index import collection_name, delete_document, ensure_collection, point_id
from src.ingestion.chunking import CHUNKER_VERSION, child_chunks, embedding_text, with_spoken_date
from src.ingestion.parsers import (
    ParsedDocument,
    ParsedSection,
    UnreadableDocument,
    parse_markdown,
    parse_pdf
)

EMBED_BATCH = 64


def index_version(resources: Resources) -> str:
    return f"{CHUNKER_VERSION}+{resources.settings.models.text_embedder.split('/')[-1]}"


def section_id(document_id: uuid.UUID, ordinal: int) -> uuid.UUID:
    return uuid.uuid5(document_id, f"section/{ordinal}")


async def _parse(resources: Resources, document: Document, data: bytes) -> ParsedDocument:
    settings = resources.settings
    match document.kind:
        case DocumentKind.PDF:
            return parse_pdf(data, document.filename, settings.storage.max_pdf_pages)
        case DocumentKind.MARKDOWN:
            return parse_markdown(data, document.filename)
        case DocumentKind.IMAGE:
            # В индекс попадает описание картинки: по нему её найдёт текстовый вопрос.
            caption = await describe_image(resources.cloud_llm, settings.llm.vision_model, data, image_mime(data))
            heading = with_spoken_date(document.filename)
            return ParsedDocument(heading, [ParsedSection("image", heading, caption)])


async def _set_status(
    resources: Resources, document_id: uuid.UUID, status: DocumentStatus, error: str | None = None
) -> None:
    async with resources.sessions.begin() as session:
        document = await session.get(Document, document_id)
        if document is not None:
            document.status = status
            document.error = error


async def ingest_document(resources: Resources, document_id: uuid.UUID) -> None:
    version = index_version(resources)
    async with resources.sessions() as session:
        document = await session.get(Document, document_id)
    if document is None:
        log.warning("ingest_skipped_missing", document_id=str(document_id))
        return
    if document.status is DocumentStatus.READY and document.index_version == version:
        log.info("ingest_skipped_ready", document_id=str(document_id))
        return

    await _set_status(resources, document_id, DocumentStatus.PROCESSING)
    try:
        data = await FileStorage(resources.settings.storage).read(document.tenant_id, document.sha256, document.kind)
        await _write(resources, document, await _parse(resources, document, data), version)
    except UnreadableDocument as error:
        # Файл не разобрать, и повтор не поможет: сообщаем причину и не перезапускаем.
        await _set_status(resources, document_id, DocumentStatus.FAILED, str(error))
        log.warning("ingest_rejected", document_id=str(document_id), reason=str(error))
    except Exception as error:
        await _set_status(resources, document_id, DocumentStatus.FAILED, f"{type(error).__name__}: {error}")
        raise


async def _write(resources: Resources, document: Document, parsed: ParsedDocument, version: str) -> None:
    sections: list[Section] = []
    ids: list[str] = []
    texts: list[str] = []
    payloads: list[dict] = []
    for ordinal, parsed_section in enumerate(parsed.sections):
        sid = section_id(document.id, ordinal)
        sections.append(
            Section(
                id=sid,
                document_id=document.id,
                tenant_id=document.tenant_id,
                ordinal=ordinal,
                content_type=parsed_section.content_type,
                page_number=parsed_section.page_number,
                heading=parsed_section.heading,
                text=parsed_section.text
            )
        )
        for index, chunk in enumerate(child_chunks(parsed_section)):
            # Название документа и раздела нужны и эмбеддингу, и реранкеру,
            # поэтому в payload лежит текст вместе с ними.
            text = embedding_text(parsed.title, parsed_section.heading, chunk)
            ids.append(point_id(document.id, ordinal, index))
            texts.append(text)
            payloads.append(
                {
                    "tenant_id": str(document.tenant_id),
                    "document_id": str(document.id),
                    "section_id": str(sid),
                    "content_type": parsed_section.content_type,
                    "chunk": text
                }
            )

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        vectors.extend(await resources.models.embed_texts(texts[start : start + EMBED_BATCH]))

    name = collection_name(resources.settings.models)
    await ensure_collection(resources.qdrant, name)
    await delete_document(resources.qdrant, name, document.tenant_id, document.id)
    await resources.qdrant.upsert(
        name,
        points=[qm.PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads, strict=True)],
        wait=True
    )

    async with resources.sessions.begin() as session:
        await session.execute(delete(Section).where(Section.document_id == document.id))
        session.add_all(sections)
        stored = await session.get(Document, document.id)
        stored.status = DocumentStatus.READY
        stored.error = None
        stored.index_version = version
    # База знаний команды изменилась: ответы в кэше, собранные без этого документа, устарели.
    await bump_kb_version(resources.redis, document.tenant_id)
    log.info("ingest_done", document_id=str(document.id), sections=len(sections), chunks=len(ids))
