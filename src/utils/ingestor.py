"""File ingestor — extracts plain text from PDF, Markdown, and plain text files.

Supports:
  - .md / .txt → direct UTF-8 read (existing behavior)
  - .pdf → layout-aware text extraction via PyMuPDF (fitz)

The output is always a plain text string suitable for injection into
PipelineState.master_resume or PipelineState.job_description.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Text sanitization (applied at ingestion time) ────────────────────
# Catches AI/Al confusion BEFORE text reaches any LLM agent
_SANITIZE_PATTERNS: list[tuple[str, str]] = [
    (r"\bAl\b(?=\s|,|$|\))", "AI"),       # "Modus Al" → "Modus AI"
    (r"\bAl/", "AI/"),                      # "Al/ML" → "AI/ML"
    (r"/Al\b", "/AI"),                      # "AI/Al" edge case
]

# Supported file extensions
_TEXT_EXTENSIONS = {".md", ".txt", ".markdown"}
_PDF_EXTENSIONS = {".pdf"}
_ALL_SUPPORTED = _TEXT_EXTENSIONS | _PDF_EXTENSIONS


class IngestionError(Exception):
    """Raised when a file cannot be ingested into usable text."""


def ingest_file(path: Path) -> str:
    """Read a file and return its content as plain text.

    Args:
        path: Path to the input file (.md, .txt, or .pdf).

    Returns:
        Extracted text content as a string.

    Raises:
        IngestionError: If the file type is unsupported, the PDF cannot be
            parsed, or the extracted text is empty (e.g., image-only PDF).
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix in _TEXT_EXTENSIONS:
        return _read_text_file(path)
    elif suffix in _PDF_EXTENSIONS:
        return _extract_pdf_text(path)
    else:
        raise IngestionError(
            f"Unsupported file type: '{suffix}'. "
            f"Supported formats: {', '.join(sorted(_ALL_SUPPORTED))}"
        )


def _read_text_file(path: Path) -> str:
    """Read a plain text or Markdown file."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise IngestionError(f"File is empty: {path}")
    logger.info("Ingested text file: %s (%d chars)", path.name, len(text))
    return text


def _sanitize_text(text: str) -> str:
    """Fix common typographic errors at ingestion time (before LLM agents see it)."""
    for pattern, replacement in _SANITIZE_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using layout-aware block sorting.

    Uses PyMuPDF's ``get_text("blocks")`` to retrieve text blocks with
    their bounding-box coordinates, then sorts by vertical position (y0)
    and horizontal position (x0) to reconstruct correct visual reading
    order. This ensures headers, contact info, and section headings are
    extracted in the order a human would read them.

    Raises IngestionError if PyMuPDF is not installed or extraction fails.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise IngestionError(
            "PyMuPDF is required for PDF ingestion. "
            "Install it with: pip install pymupdf"
        )

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise IngestionError(
            f"Failed to open PDF: {path}\n"
            f"Error: {exc}\n\n"
            "If this is a corrupted or encrypted PDF, please provide "
            "a plain text (.md or .txt) version instead."
        ) from exc

    pages_text: list[str] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = _extract_page_blocks(page)
            # Reconstruct hyperlinks from PDF annotations on this page
            page_text = _restore_pdf_links(page, page_text)
            if page_text.strip():
                pages_text.append(page_text.strip())
    finally:
        doc.close()

    if not pages_text:
        raise IngestionError(
            f"No extractable text found in PDF: {path}\n\n"
            "This is likely a scanned/image-based PDF without embedded text. "
            "Please provide a text-based version (.md or .txt) instead, or "
            "re-export the PDF with selectable text enabled."
        )

    full_text = "\n\n".join(pages_text)

    # Sanitize at ingestion time so agents never see "Al" for "AI"
    full_text = _sanitize_text(full_text)

    logger.info(
        "Ingested PDF: %s (%d pages, %d chars)",
        path.name,
        len(pages_text),
        len(full_text),
    )
    return full_text


def _restore_pdf_links(page, text: str) -> str:
    """Reconstruct markdown-style links from PDF hyperlink annotations.

    PyMuPDF's get_text() strips hyperlink URLs, leaving only the visible
    label text (e.g., "LinkedIn" instead of "[LinkedIn](https://...)").
    This function reads the page's link annotations, finds the display
    text at each link's bounding box, and replaces the plain text with
    a markdown link so the downstream formatter can render it as a
    clickable <a href>.
    """
    try:
        links = page.get_links()
    except Exception:
        return text

    if not links:
        return text

    # Collect (label, url) pairs — only URI links with real URLs
    restorations: list[tuple[str, str]] = []
    for link in links:
        if link.get("kind") != 2:  # kind 2 = URI link
            continue
        uri = link.get("uri", "").strip()
        if not uri:
            continue
        # Extract the display text at this link's bounding box
        rect = link.get("from")
        if rect is None:
            continue
        try:
            label = page.get_text("text", clip=rect).strip()
        except Exception:
            continue
        if label:
            restorations.append((label, uri))

    # Apply restorations — replace plain label with [label](url)
    # Process longest labels first to avoid partial matches
    restorations.sort(key=lambda pair: len(pair[0]), reverse=True)
    for label, uri in restorations:
        md_link = f"[{label}]({uri})"
        # Only replace standalone occurrences (not already inside markdown links)
        if f"[{label}](" not in text:
            text = text.replace(label, md_link, 1)
            logger.debug("Restored PDF link: %s → %s", label, md_link)

    return text


def _extract_page_blocks(page) -> str:
    """Extract text from a single page using layout-aware block sorting.

    PyMuPDF's get_text("blocks") returns tuples:
        (x0, y0, x1, y1, text, block_no, block_type)
    where block_type 0 = text, 1 = image.

    We filter to text blocks, sort by (y0, x0) for top-to-bottom,
    left-to-right reading order, then join the text.
    """
    try:
        blocks = page.get_text("blocks")
    except Exception:
        # Graceful fallback — if block extraction fails, try plain text
        logger.warning("Block extraction failed for page, falling back to plain text")
        return page.get_text("text")

    if not blocks:
        return ""

    # Filter to text blocks only (block_type == 0)
    text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]

    if not text_blocks:
        return ""

    # Sort by vertical position (y0), then horizontal (x0)
    text_blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))

    # Extract and join text from each block
    parts: list[str] = []
    for block in text_blocks:
        text = block[4]
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())

    return "\n".join(parts)
