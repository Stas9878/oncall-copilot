"""Хранилище загруженных файлов."""

import uuid
import anyio
from pathlib import PurePosixPath

from src.db.models import DocumentKind
from src.core.config import StorageSettings

_SUFFIX = {DocumentKind.PDF: ".pdf", DocumentKind.MARKDOWN: ".md", DocumentKind.IMAGE: ".img"}


class FileStorage:
    def __init__(self, settings: StorageSettings) -> None:
        self._root = anyio.Path(settings.uploads_dir)

    def _path(self, tenant_id: uuid.UUID, sha256: str, kind: DocumentKind) -> anyio.Path:
        # Имя файла — хеш содержимого, а не имя от пользователя: путь нельзя
        # подменить строкой вида ../../etc/passwd.
        return self._root / str(tenant_id) / PurePosixPath(sha256 + _SUFFIX[kind]).name

    async def save(self, tenant_id: uuid.UUID, sha256: str, kind: DocumentKind, data: bytes) -> None:
        path = self._path(tenant_id, sha256, kind)
        await path.parent.mkdir(parents=True, exist_ok=True)
        await path.write_bytes(data)

    async def read(self, tenant_id: uuid.UUID, sha256: str, kind: DocumentKind) -> bytes:
        return await self._path(tenant_id, sha256, kind).read_bytes()

    async def delete(self, tenant_id: uuid.UUID, sha256: str, kind: DocumentKind) -> None:
        await self._path(tenant_id, sha256, kind).unlink(missing_ok=True)
