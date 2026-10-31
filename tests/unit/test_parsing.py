from pathlib import Path

from src.ingestion.parsers import parse_markdown, parse_pdf
from src.ingestion.chunking import child_chunks, with_spoken_date

RUNBOOK = Path("corpus/kira/runbook_dispatch.pdf").read_bytes()


def test_table_rows_keep_underscores():
    document = parse_pdf(RUNBOOK, "runbook_dispatch.pdf", max_pages=50)
    rows = [row for section in document.sections for row in section.rows]
    assert any("DISPATCH_READ_PRIMARY=true" in row for row in rows)


def test_scanned_page_goes_through_ocr():
    document = parse_pdf(RUNBOOK, "runbook_dispatch.pdf", max_pages=50)
    [scan] = [section for section in document.sections if section.content_type == "ocr"]
    assert "вечерним пиком" in scan.heading
    assert "квоты геокодера" in scan.text


def test_table_row_is_one_chunk():
    document = parse_pdf(RUNBOOK, "runbook_dispatch.pdf", max_pages=50)
    [table] = [section for section in document.sections if section.content_type == "table"]
    assert child_chunks(table) == table.rows


def test_markdown_split_by_headings():
    document = parse_markdown(Path("corpus/lumen/oncall_policy_lumen.md").read_bytes(), "policy.md")
    assert document.title == "Политика дежурств платёжной команды Lumen Pay"
    assert [s.heading for s in document.sections] == [
        "Уровни серьёзности",
        "Обязанности дежурного",
        "Эскалация",
        "Внешние ресурсы"
    ]


def test_date_in_filename_spelled_out():
    assert with_spoken_date("grafana_payments_2026-09-18.png") == "grafana_payments_2026-09-18.png, 18 сентября 2026"
    assert with_spoken_date("architecture.png") == "architecture.png"
