"""Formatter Agent — renders the approved draft as a polished single-page PDF.

Purely programmatic (no LLM call). Uses ReportLab to generate a
professionally styled PDF with:
  - Flush-right date alignment (baseline-matched via same-size table cells)
  - Clickable hyperlinks for valid URLs
  - Education-specific layout (org on own line, degree below)
  - Text sanitization (AI vs Al, name cleaning)
  - Tight spacing optimized for single-page density
"""

from __future__ import annotations

import logging
import re
from io import BytesIO
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..state import PipelineState, ResumeSection

logger = logging.getLogger(__name__)

# ── Style constants ──────────────────────────────────────────────
FONT_NAME = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_BOLD_ITALIC = "Helvetica-BoldOblique"
COLOR_HEADING = HexColor("#1A1A1A")
COLOR_BODY = HexColor("#2D2D2D")
COLOR_SUBTLE = HexColor("#4A4A4A")
COLOR_BULLET = HexColor("#777777")
COLOR_LINK = HexColor("#1A5276")
COLOR_RULE = HexColor("#3B3B3B")

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_LEFT = 0.5 * inch
MARGIN_RIGHT = 0.5 * inch
MARGIN_TOP = 0.4 * inch
MARGIN_BOTTOM = 0.4 * inch
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT

# ── Text sanitization patterns ───────────────────────────────────
# Common LLM font-confusion fixes (sans-serif I vs l)
SANITIZE_PATTERNS: list[tuple[str, str]] = [
    (r"\bAl\b(?=\s|,|$|\))", "AI"),       # "Modus Al" → "Modus AI"
    (r"\bAl/", "AI/"),                      # "Al/ML" → "AI/ML"
    (r"/Al\b", "/AI"),                      # "AI/Al" edge case
]


def _sanitize_text(text: str) -> str:
    """Fix common LLM typographic errors before PDF rendering."""
    for pattern, replacement in SANITIZE_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


# ── Styles ───────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    """Paragraph styles optimized for single-page density."""
    base = getSampleStyleSheet()

    # Shared leading for title/date rows — ensures baseline alignment
    ENTRY_SIZE = 9.5
    ENTRY_LEADING = 12

    return {
        "name": ParagraphStyle(
            "ResumeName",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=17,
            leading=20,
            alignment=TA_CENTER,
            textColor=COLOR_HEADING,
            spaceAfter=1,
            spaceBefore=0,
        ),
        "contact": ParagraphStyle(
            "ResumeContact",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=COLOR_SUBTLE,
            spaceAfter=2,
            spaceBefore=0,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=10,
            leading=13,
            textColor=COLOR_HEADING,
            spaceBefore=7,
            spaceAfter=1,
        ),
        # ── Entry title row: left column ─────────────────────────
        "entry_left": ParagraphStyle(
            "EntryLeft",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=ENTRY_SIZE,
            leading=ENTRY_LEADING,
            textColor=COLOR_HEADING,
        ),
        # ── Entry title row: right column (dates) ────────────────
        "entry_dates": ParagraphStyle(
            "EntryDates",
            parent=base["Normal"],
            fontName=FONT_ITALIC,
            fontSize=ENTRY_SIZE,
            leading=ENTRY_LEADING,
            textColor=COLOR_SUBTLE,
            alignment=TA_RIGHT,
        ),
        # ── Education: degree line below org ─────────────────────
        "edu_degree": ParagraphStyle(
            "EduDegree",
            parent=base["Normal"],
            fontName=FONT_ITALIC,
            fontSize=9,
            leading=11.5,
            textColor=COLOR_BODY,
            leftIndent=0,
            spaceAfter=1,
        ),
        # ── Bullet points ────────────────────────────────────────
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=11.5,
            textColor=COLOR_BODY,
            leftIndent=11,
            firstLineIndent=-11,
            spaceBefore=0.5,
            spaceAfter=2.5,
        ),
        # ── Skills lines ─────────────────────────────────────────
        "skills": ParagraphStyle(
            "Skills",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=11.5,
            textColor=COLOR_BODY,
            leftIndent=4,
            spaceAfter=1.5,
        ),
    }


# ── The Agent ────────────────────────────────────────────────────

class FormatterAgent:
    """Renders pruned_sections + contact info into a polished single-page PDF."""

    def __init__(self) -> None:
        self.styles = _build_styles()

    # ── Public API ───────────────────────────────────────────────

    def test_render(self, state: PipelineState) -> int:
        """Render to an in-memory buffer and return the page count."""
        buffer = BytesIO()
        page_count = self._build_pdf(state, buffer)
        buffer.close()
        logger.info("  Test render: %d page(s)", page_count)
        return page_count

    def run(self, state: PipelineState, output_path: Path) -> PipelineState:
        """Build and save the final PDF resume."""
        logger.info("▶ Running formatter agent [reportlab — no LLM call]")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        page_count = self._build_pdf(state, str(output_path))
        logger.info(
            "✔ formatter agent complete — %d page(s), saved to %s",
            page_count,
            output_path,
        )
        return state.model_copy(update={"final_resume": str(output_path)})

    # ── PDF builder (shared by test_render and run) ──────────────

    def _build_pdf(self, state: PipelineState, output) -> int:
        """Build the PDF to a file path or BytesIO buffer. Returns page count."""
        doc = SimpleDocTemplate(
            output,
            pagesize=letter,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        story: list = []

        # ── Contact header ───────────────────────────────────────
        name, contact_html = self._extract_contact(state.master_resume)
        story.append(Paragraph(_sanitize_text(self._esc(name)), self.styles["name"]))
        if contact_html:
            story.append(Paragraph(contact_html, self.styles["contact"]))

        # ── Resume sections ──────────────────────────────────────
        if state.pruned_sections:
            for section in state.pruned_sections:
                story.extend(self._build_section(section))

        # ── Build and capture page count ─────────────────────────
        page_counter = _PageCounter()
        doc.build(
            story,
            onFirstPage=page_counter.on_page,
            onLaterPages=page_counter.on_page,
        )
        return page_counter.count

    # ── Contact extraction ───────────────────────────────────────

    @staticmethod
    def _extract_contact(master_resume: str) -> tuple[str, str]:
        """Extract name and contact HTML (with clickable links) from header."""
        name = "Resume"
        contact_line = ""
        if not master_resume:
            return name, contact_line

        for line in master_resume.split("\n"):
            stripped = line.strip()
            if stripped.startswith("---"):
                break
            # First markdown heading → name
            if stripped.startswith("#") and name == "Resume":
                name = stripped.lstrip("#").strip()
                name = re.sub(r"[*_]", "", name).strip()
                # Strip descriptors: "— Master Resume", "- Resume", etc.
                name = re.sub(
                    r"\s*[—–\-]\s*(Master\s+)?Resume.*$",
                    "",
                    name,
                    flags=re.IGNORECASE,
                ).strip()
            # Contact info line (has | or @)
            elif ("|" in stripped or "@" in stripped) and name != "Resume":
                contact_line = re.sub(r"\*\*|\*|__|_", "", stripped)

                # Convert markdown links → ReportLab <a> hyperlinks
                def _link_repl(m: re.Match) -> str:
                    text, url = m.group(1), m.group(2)
                    if url and url != "#" and url.startswith(("http://", "https://")):
                        return (
                            f'<a href="{url}" color="#1A5276">'
                            f"<u>{text}</u></a>"
                        )
                    return text

                contact_line = re.sub(
                    r"\[([^\]]+)\]\(([^)]*)\)", _link_repl, contact_line
                )

                # Make bare email addresses clickable
                contact_line = re.sub(
                    r"([\w.+-]+@[\w-]+\.[\w.-]+)",
                    r'<a href="mailto:\1" color="#1A5276">\1</a>',
                    contact_line,
                )

                contact_line = contact_line.strip()

        return name, contact_line

    # ── Section builder ──────────────────────────────────────────

    def _build_section(self, section: ResumeSection) -> list:
        """Build flowables for one resume section."""
        elements: list = []
        heading = section.heading.upper()

        elements.append(
            Paragraph(self._esc(heading), self.styles["section_heading"])
        )
        elements.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=COLOR_RULE,
                spaceAfter=3,
                spaceBefore=0,
            )
        )

        is_skills = "skill" in section.heading.lower()
        is_education = "edu" in section.heading.lower()

        for entry in section.entries:
            if is_skills:
                elements.extend(self._build_skills_entry(entry))
            elif is_education:
                elements.extend(self._build_education_entry(entry))
            else:
                elements.extend(self._build_standard_entry(entry))

        return elements

    # ── Education entry (special layout) ─────────────────────────

    def _build_education_entry(self, entry) -> list:
        """Education: org on line 1 with dates flush-right, degree on line 2."""
        elements: list = []

        # Line 1: University name (left, bold) | Dates (right, italic)
        org_para = Paragraph(
            f"<b>{self._esc(entry.organization)}</b>",
            self.styles["entry_left"],
        )
        dates_para = Paragraph(
            self._esc(entry.dates) if entry.dates else "",
            self.styles["entry_dates"],
        )
        tbl = self._make_entry_row(org_para, dates_para, space_before=4)
        elements.append(tbl)

        # Line 2: Degree / title
        if entry.title:
            elements.append(
                Paragraph(
                    _sanitize_text(self._esc(entry.title)),
                    self.styles["edu_degree"],
                )
            )

        # Bullets
        for bullet in entry.bullets:
            elements.append(self._make_bullet(bullet))

        return elements

    # ── Standard entry (work experience, projects) ───────────────

    def _build_standard_entry(self, entry) -> list:
        """Title | Org (left), dates flush-right (same baseline)."""
        elements: list = []

        # Left: bold title + org
        left_parts = [f"<b>{_sanitize_text(self._esc(entry.title))}</b>"]
        if entry.organization:
            left_parts.append(
                f'<font color="{COLOR_SUBTLE.hexval()}">'
                f"  |  </font>{_sanitize_text(self._esc(entry.organization))}"
            )
        left_para = Paragraph("".join(left_parts), self.styles["entry_left"])

        # Right: dates
        dates_para = Paragraph(
            self._esc(entry.dates) if entry.dates else "",
            self.styles["entry_dates"],
        )

        tbl = self._make_entry_row(left_para, dates_para, space_before=4)
        elements.append(tbl)

        # Bullets
        for bullet in entry.bullets:
            elements.append(self._make_bullet(bullet))

        return elements

    # ── Skills entry ─────────────────────────────────────────────

    def _build_skills_entry(self, entry) -> list:
        """Render skills as category: items lines."""
        elements: list = []
        for bullet in entry.bullets:
            text = _sanitize_text(bullet)
            if ":" in text:
                cat, items = text.split(":", 1)
                markup = f"<b>{self._esc(cat.strip())}:</b> {self._esc(items.strip())}"
            else:
                markup = self._esc(text)
            elements.append(Paragraph(markup, self.styles["skills"]))
        return elements

    # ── Shared helpers ───────────────────────────────────────────

    def _make_entry_row(self, left: Paragraph, right: Paragraph, space_before: float = 4) -> Table:
        """Two-column table for baseline-aligned title + dates rows."""
        col_widths = [CONTENT_WIDTH * 0.76, CONTENT_WIDTH * 0.24]
        tbl = Table([[left, right]], colWidths=col_widths)
        tbl.setStyle(
            TableStyle(
                [
                    # BOTTOM valign ensures baselines align across columns
                    ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), space_before),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        return tbl

    def _make_bullet(self, text: str) -> Paragraph:
        """Create a single bullet-point paragraph."""
        clean = _sanitize_text(text)
        markup = (
            f'<font color="{COLOR_BULLET.hexval()}">\u2022</font>'
            f"  {self._esc(clean)}"
        )
        return Paragraph(markup, self.styles["bullet"])

    @staticmethod
    def _esc(text: str) -> str:
        """Escape XML-special characters for ReportLab Paragraph markup."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )


# ── Page counter callback ────────────────────────────────────────

class _PageCounter:
    """Tracks total page count during a ReportLab build."""

    def __init__(self) -> None:
        self.count = 0

    def on_page(self, canvas, doc) -> None:
        self.count = doc.page
