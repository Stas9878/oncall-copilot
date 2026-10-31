"""Разбор файлов на разделы."""

import re
import pymupdf
from dataclasses import dataclass, field

# Совет поставить pymupdf_layout печатается в stdout и ломает JSON-логи.
pymupdf.no_recommend_layout()

# Страница, где текстового слоя меньше, считается сканом и идёт в OCR.
MIN_NATIVE_CHARS = 40
HEADING_SIZE_RATIO = 1.15
BOLD_FLAG = 16


@dataclass(slots=True)
class ParsedSection:
    content_type: str  # text, table, ocr, markdown или image
    heading: str | None
    text: str
    page_number: int | None = None
    # Строки таблицы: каждая станет отдельным фрагментом для поиска.
    rows: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    sections: list[ParsedSection]


class UnreadableDocument(ValueError):
    """Файл не удалось разобрать: битый PDF или пустой текст."""


def _join_lines(lines: list[str]) -> str:
    # PDF переносит строки внутри абзаца. Перенос после дефиса склеиваем без
    # пробела: «мерчант-» + «аккаунт» -> «мерчант-аккаунт».
    text = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not text:
            text = line
        elif text.endswith("-"):
            text += line
        else:
            text += ("\n" if line.startswith("•") else " ") + line
    return text


def _cell_text(page: pymupdf.Page, bbox: tuple | None) -> str:
    # Table.extract() теряет подчёркивания: DISPATCH_READ_PRIMARY превращается в
    # «DISPATCH READ PRIMARY». Текст по рамке ячейки сохраняет символы как есть.
    if bbox is None:
        return ""
    return " ".join(page.get_text("text", clip=pymupdf.Rect(bbox)).split())


def _table_rows(page: pymupdf.Page, table: pymupdf.table.Table) -> tuple[str, list[str]]:
    header, *body = [[_cell_text(page, cell) for cell in row.cells] for row in table.rows]
    markdown = "\n".join(
        ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
        + ["| " + " | ".join(row) + " |" for row in body]
    )
    # «Алерт: X | Первое действие: Y» находится поиском лучше, чем голая строка значений.
    readable = [
        " | ".join(f"{name}: {value}" for name, value in zip(header, row, strict=False) if value) for row in body
    ]
    return markdown, readable


def _ocr_section(page: pymupdf.Page) -> ParsedSection | None:
    textpage = page.get_textpage_ocr(language="rus+eng", dpi=300, full=True)
    lines = [line.strip() for line in page.get_text("text", textpage=textpage).splitlines() if line.strip()]
    if not lines:
        return None
    return ParsedSection("ocr", lines[0], "\n".join(lines[1:]) or lines[0], page.number + 1)


def _page_sections(
    page: pymupdf.Page, body_size: float, current_heading: str | None
) -> tuple[list[ParsedSection], str | None]:
    tables = [table for table in page.find_tables().tables if table.row_count >= 2]
    table_rects = [pymupdf.Rect(table.bbox) for table in tables]
    sections: list[ParsedSection] = []
    heading, lines = current_heading, []

    def flush() -> None:
        if lines:
            sections.append(ParsedSection("text", heading, _join_lines(lines), page.number + 1))
            lines.clear()

    for block in page.get_text("dict", sort=True)["blocks"]:
        for line in block.get("lines", []):
            rect = pymupdf.Rect(line["bbox"])
            if any(table_rect.intersects(rect) for table_rect in table_rects):
                continue
            span = line["spans"][0]
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            if span["flags"] & BOLD_FLAG and span["size"] >= body_size * HEADING_SIZE_RATIO:
                flush()
                heading = text
            else:
                lines.append(text)
    flush()

    for table in tables:
        markdown, rows = _table_rows(page, table)
        sections.append(ParsedSection("table", heading, markdown, page.number + 1, rows))
    return sections, heading


def _body_font_size(document: pymupdf.Document) -> float:
    sizes: dict[float, int] = {}
    for page in document:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    sizes[size] = sizes.get(size, 0) + len(span["text"])
    return max(sizes, key=sizes.get) if sizes else 10.0


def parse_pdf(data: bytes, filename: str, max_pages: int) -> ParsedDocument:
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise UnreadableDocument(f"PDF не открывается: {error}") from error
    with document:
        if document.page_count > max_pages:
            raise UnreadableDocument(f"В PDF {document.page_count} страниц, лимит {max_pages}")
        body_size = _body_font_size(document)
        sections: list[ParsedSection] = []
        heading: str | None = None
        for page in document:
            if len(page.get_text("text").strip()) < MIN_NATIVE_CHARS and page.get_images():
                section = _ocr_section(page)
                sections.extend([section] if section else [])
                continue
            page_sections, heading = _page_sections(page, body_size, heading)
            sections.extend(page_sections)
        title = _pdf_title(document[0]) or filename
    if not sections:
        raise UnreadableDocument("В PDF не найдено текста")
    return ParsedDocument(title, sections)


def _pdf_title(page: pymupdf.Page) -> str | None:
    """Название документа: строка самым крупным шрифтом на первой странице."""
    best_size, best_text = 0.0, None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line["spans"]).strip()
            size = line["spans"][0]["size"]
            if text and size > best_size:
                best_size, best_text = size, text
    return best_text


_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")


def parse_markdown(data: bytes, filename: str) -> ParsedDocument:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise UnreadableDocument("Markdown должен быть в UTF-8") from error
    title, heading, lines = filename, None, []
    sections: list[ParsedSection] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            sections.append(ParsedSection("markdown", heading, body))
        lines.clear()

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match and len(match.group(1)) == 1:
            title = match.group(2).strip()
            heading = title
        elif match:
            flush()
            heading = match.group(2).strip()
        else:
            lines.append(line)
    flush()
    if not sections:
        raise UnreadableDocument("В Markdown нет текста")
    return ParsedDocument(title, sections)
