"""Загрузка документов в базу знаний команды и статус их индексации."""

import uuid
import hashlib
from datetime import datetime
from sqlalchemy import select
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, Response, UploadFile, status

from src.core.storage import FileStorage
from src.worker import ingest_document_task
from src.security.uploads import detect_kind, read_limited
from src.search.index import collection_name, delete_document
from src.db.models import Document, DocumentKind, DocumentStatus
from src.api.deps import EditorDep, PrincipalDep, ResourcesDep, SessionDep

router = APIRouter(prefix="/v1/documents", tags=["documents"])


class DocumentOut(BaseModel):
    id: uuid.UUID
    filename: str
    kind: DocumentKind
    status: DocumentStatus
    error: str | None
    created_at: datetime

    @classmethod
    def of(cls, document: Document) -> DocumentOut:
        return cls.model_validate(document, from_attributes=True)


async def _get_own(session: AsyncSession, tenant_id: uuid.UUID, document_id: uuid.UUID) -> Document:
    document = await session.get(Document, document_id)
    # Чужой документ отвечает тем же 404, что и несуществующий:
    # по ответу нельзя узнать, что такой id есть у другой команды.
    if document is None or document.tenant_id != tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Документ не найден")
    return document


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=DocumentOut)
async def upload(
    file: UploadFile, principal: EditorDep, session: SessionDep, resources: ResourcesDep, response: Response
) -> DocumentOut:
    """Принимает файл и ставит индексацию в очередь. Ответ не ждёт индексации."""
    storage_settings = resources.settings.storage
    data = await read_limited(file, storage_settings.max_upload_mb * 1024 * 1024)
    filename = (file.filename or "upload").rsplit("/", 1)[-1][:255]
    kind = detect_kind(data, filename)
    sha256 = hashlib.sha256(data).hexdigest()

    existing = await session.scalar(
        select(Document).where(Document.tenant_id == principal.tenant_id, Document.sha256 == sha256)
    )
    if existing is not None:
        # Тот же файл уже загружен: повторная загрузка ничего не меняет.
        response.status_code = status.HTTP_200_OK
        return DocumentOut.of(existing)

    await FileStorage(storage_settings).save(principal.tenant_id, sha256, kind, data)
    document = Document(
        tenant_id=principal.tenant_id, filename=filename, kind=kind, sha256=sha256, size_bytes=len(data)
    )
    session.add(document)
    # Сначала commit, потом задача: иначе воркер может взять задачу раньше,
    # чем строка документа появится в базе.
    await session.commit()
    await ingest_document_task.kiq(str(document.id))
    return DocumentOut.of(document)


@router.get("", response_model=list[DocumentOut])
async def list_documents(principal: PrincipalDep, session: SessionDep) -> list[DocumentOut]:
    documents = await session.scalars(
        select(Document).where(Document.tenant_id == principal.tenant_id).order_by(Document.created_at.desc())
    )
    return [DocumentOut.of(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: uuid.UUID, principal: PrincipalDep, session: SessionDep) -> DocumentOut:
    return DocumentOut.of(await _get_own(session, principal.tenant_id, document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(document_id: uuid.UUID, principal: EditorDep, session: SessionDep, resources: ResourcesDep) -> None:
    document = await _get_own(session, principal.tenant_id, document_id)
    await delete_document(
        resources.qdrant, collection_name(resources.settings.models), principal.tenant_id, document.id
    )
    await session.delete(document)
    await session.commit()
    await FileStorage(resources.settings.storage).delete(principal.tenant_id, document.sha256, document.kind)
