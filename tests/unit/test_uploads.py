import pytest
from fastapi import HTTPException

from src.db.models import DocumentKind
from src.security.uploads import detect_kind

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_type_detected_by_content_not_extension():
    assert detect_kind(b"%PDF-1.7 ...", "notes.md") is DocumentKind.PDF
    assert detect_kind(PNG, "report.pdf") is DocumentKind.IMAGE
    assert detect_kind("# Ранбук".encode(), "runbook.md") is DocumentKind.MARKDOWN


def test_unknown_content_rejected():
    with pytest.raises(HTTPException) as error:
        detect_kind(b"MZ\x90\x00 executable", "runbook.pdf")
    assert error.value.status_code == 415
