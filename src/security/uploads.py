"""Проверка загружаемых файлов до записи на диск."""

import base64
import binascii
from fastapi import HTTPException, UploadFile, status

from src.db.models import DocumentKind
from src.ingestion.captions import IMAGE_SIGNATURES

READ_CHUNK = 1024 * 1024


async def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Читает файл частями и прекращает чтение на лимите, не дожидаясь конца файла."""
    data = bytearray()
    while chunk := await upload.read(READ_CHUNK):
        data.extend(chunk)
        if len(data) > max_bytes:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"Файл больше {max_bytes // (1024 * 1024)} МБ")
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "Пустой файл")
    return bytes(data)


def detect_kind(data: bytes, filename: str) -> DocumentKind:
    """Тип файла по содержимому, а не по расширению или Content-Type от клиента."""
    if data.startswith(b"%PDF-"):
        return DocumentKind.PDF
    if data.startswith(tuple(IMAGE_SIGNATURES)):
        return DocumentKind.IMAGE
    if filename.lower().endswith((".md", ".markdown")):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        else:
            return DocumentKind.MARKDOWN
    raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Поддерживаются PDF, PNG, JPEG и Markdown в UTF-8")


MAX_SCREEN_BYTES = 5 * 1024 * 1024


def decode_screen(image_base64: str) -> bytes:
    """Скрин из JSON: base64, не больше 5 МБ, PNG или JPEG по сигнатуре."""
    try:
        image = base64.b64decode(image_base64, validate=True)
    except binascii.Error as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "image_base64 не декодируется") from error
    if len(image) > MAX_SCREEN_BYTES:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "Скрин больше 5 МБ")
    if not image.startswith(tuple(IMAGE_SIGNATURES)):
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Скрин должен быть PNG или JPEG")
    return image
