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

from dataclasses import dataclass
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
AVAILABLE_HEIGHT = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

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
    base["Normal"].leading = 14

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
            spaceBefore=14,
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
            leading=13,
            textColor=COLOR_BODY,
            leftIndent=0,
            spaceAfter=1,
        ),
        # ── Bullet points (hanging indent: wrapped lines align under text, not bullet glyph)
        # leftIndent=24 nests bullets visually under section headings;
        # firstLineIndent=-12 hangs the bullet glyph to the left of the text.
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=14,
            textColor=COLOR_BODY,
            leftIndent=24,
            firstLineIndent=-12,
            spaceBefore=0.5,
            spaceAfter=2,
        ),
        # ── Skills lines ─────────────────────────────────────────
        "skills": ParagraphStyle(
            "Skills",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=13,
            textColor=COLOR_BODY,
            leftIndent=4,
            spaceAfter=1.5,
        ),
    }


@dataclass(frozen=True)
class RenderMetrics:
    """Layout metrics captured from a formatter test render."""

    page_count: int
    fill_ratio: float
    used_height: float
    available_height: float


# ── The Agent ────────────────────────────────────────────────────

class FormatterAgent:
    """Renders pruned_sections + contact info into a polished single-page PDF."""

    def __init__(self) -> None:
        self.styles = _build_styles()

    # ── Public API ───────────────────────────────────────────────

    def test_render(self, state: PipelineState) -> int:
        """Render to an in-memory buffer and return the page count."""
        return self.measure_render(state).page_count

    def measure_render(self, state: PipelineState) -> RenderMetrics:
        """Render in-memory and capture page count plus estimated fill ratio.

        Builds the story once and reuses it for both height measurement and
        PDF render (previously built it twice per call).
        """
        story = self._build_story(state)
        used_height = self._measure_story_height(story)

        buffer = BytesIO()
        page_count = self._build_story_to_output(story, buffer)
        buffer.close()

        fill_ratio = used_height / AVAILABLE_HEIGHT if AVAILABLE_HEIGHT else 0.0
        metrics = RenderMetrics(
            page_count=page_count,
            fill_ratio=fill_ratio,
            used_height=used_height,
            available_height=AVAILABLE_HEIGHT,
        )
        logger.info(
            "  Test render: %d page(s), fill %.0f%%",
            metrics.page_count,
            metrics.fill_ratio * 100,
        )
        return metrics

    def measure_fill_ratio(self, state: PipelineState) -> float:
        """Return how much of the usable page height the content fills (0.0–1.0).

        Builds the flowable story without rendering to PDF, then sums the
        wrapped heights to estimate fill ratio against the available frame.
        """
        available_height = PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

        story: list = []
        name, contact_html = self._extract_contact(state.master_resume)
        story.append(Paragraph(_sanitize_text(self._esc(name)), self.styles["name"]))
        if contact_html:
            story.append(Paragraph(contact_html, self.styles["contact"]))
        if state.pruned_sections:
            for section in state.pruned_sections:
                story.extend(self._build_section(section))

        # Measure each flowable's wrapped height
        total_height = 0.0
        for flowable in story:
            w, h = flowable.wrap(CONTENT_WIDTH, available_height)
            total_height += h
            # Account for spaceBefore/spaceAfter on Paragraphs
            if hasattr(flowable, 'style'):
                total_height += getattr(flowable.style, 'spaceBefore', 0)
                total_height += getattr(flowable.style, 'spaceAfter', 0)

        ratio = min(total_height / available_height, 1.0)
        logger.info("  Content fill ratio: %.1f%% (%d/%dpt)", ratio * 100, total_height, available_height)
        return ratio

    def run(self, state: PipelineState, output_path: Path) -> PipelineState:
        """Build and save the final PDF resume."""
        logger.info("▶ Running formatter agent [reportlab — no LLM call]")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        buffer = BytesIO()
        page_count = self._build_pdf(state, buffer)
        if page_count > 1:
            buffer.close()
            raise RuntimeError(
                f"Formatter refused to write final PDF because it rendered to {page_count} pages."
            )

        output_path.write_bytes(buffer.getvalue())
        buffer.close()
        logger.info(
            "✔ formatter agent complete — %d page(s), saved to %s",
            page_count,
            output_path,
        )
        return state.model_copy(update={"final_resume": str(output_path)})

    # ── PDF builder (shared by test_render and run) ──────────────

    def _build_pdf(self, state: PipelineState, output) -> int:
        """Build the PDF to a file path or BytesIO buffer. Returns page count."""
        story = self._build_story(state)
        return self._build_story_to_output(story, output)

    def _build_story(self, state: PipelineState) -> list:
        """Build the ordered flowables used for both render and measurement."""
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

        return story

    def _build_story_to_output(self, story: list, output) -> int:
        """Build a preconstructed story to a file path or BytesIO buffer."""
        doc = SimpleDocTemplate(
            output,
            pagesize=letter,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        # ── Build and capture page count ─────────────────────────
        page_counter = _PageCounter()
        doc.build(
            story,
            onFirstPage=page_counter.on_page,
            onLaterPages=page_counter.on_page,
        )
        return page_counter.count

    def _measure_story_height(self, story: list) -> float:
        """Estimate consumed vertical space using ReportLab flowable heights."""
        used_height = 0.0
        for flowable in story:
            width, height = flowable.wrap(CONTENT_WIDTH, AVAILABLE_HEIGHT)
            del width
            used_height += float(height)
            if hasattr(flowable, "getSpaceBefore"):
                used_height += float(flowable.getSpaceBefore())
            if hasattr(flowable, "getSpaceAfter"):
                used_height += float(flowable.getSpaceAfter())
        return used_height

    # ── Contact extraction ───────────────────────────────────────

    @staticmethod
    def _extract_contact(master_resume: str) -> tuple[str, str]:
        """Deterministic regex extraction of name and contact HTML.

        Bypasses LLM entirely. Parses the raw master_resume text using
        pure Python regex to find name, email, phone, and markdown links.
        Handles both Markdown (# heading) and plain text (PDF) formats.
        """
        if not master_resume:
            return "Resume", ""

        name = ""
        contact_html = ""

        for line in master_resume.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("---"):
                break

            # ── Name: first heading or first non-contact line ─────
            if not name:
                if stripped.startswith("#"):
                    name = stripped.lstrip("#").strip()
                elif "|" not in stripped and "@" not in stripped:
                    name = stripped
                # Clean: strip markdown, "— Master Resume", etc.
                name = re.sub(r"[*_]", "", name).strip()
                name = re.sub(
                    r"\s*[—–\-]\s*(Master\s+)?Resume.*$", "",
                    name, flags=re.IGNORECASE,
                ).strip()
                if "|" not in stripped and "@" not in stripped:
                    continue

            # ── Contact: line with | or @ ─────────────────────────
            if ("|" in stripped or "@" in stripped) and not contact_html:
                raw = re.sub(r"\*\*|\*|__|_", "", stripped)

                # Build contact parts deterministically
                parts: list[str] = []
                for segment in raw.split("|"):
                    segment = segment.strip()
                    if not segment:
                        continue

                    # 1) Markdown link: [Text](URL)
                    md_match = re.match(
                        r'\[([^\]]+)\]\(([^)]+)\)', segment
                    )
                    if md_match:
                        label, url = md_match.group(1), md_match.group(2).strip()
                        if url and url != "#":
                            # Ensure URL has a scheme
                            if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", url):
                                url = f"https://{url.lstrip('/')}"
                            parts.append(
                                f'<a href="{url}" color="blue"><u>{label}</u></a>'
                            )
                        else:
                            # Placeholder link (#) — render as plain text
                            parts.append(segment.replace("[", "").split("]")[0])
                        continue

                    # 2) Bare email address
                    email_match = re.match(
                        r"([\w.+-]+@[\w-]+\.[\w.-]+)$", segment
                    )
                    if email_match:
                        email = email_match.group(1)
                        parts.append(
                            f'<a href="mailto:{email}" color="blue">'
                            f"<u>{email}</u></a>"
                        )
                        continue

                    # 3) Plain text (city, phone, etc.)
                    parts.append(segment)

                contact_html = "  |  ".join(parts)

            if name and contact_html:
                break

        return name or "Resume", contact_html

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
                spaceAfter=6,
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
        """Education: org on line 1 with dates flush-right, degree on line 2.

        The Mapper may place the university name in either `organization` or
        `title`. We detect which field holds the university by checking for
        keywords like "university", "college", "institute", then assign
        accordingly so the bold line is always the institution and the italic
        line is always the degree.
        """
        elements: list = []

        # ── Normalize field assignment ────────────────────────────
        org_text = (entry.organization or "").strip()
        title_text = (entry.title or "").strip()

        uni_keywords = ("university", "college", "institute", "school")
        org_is_uni = any(kw in org_text.lower() for kw in uni_keywords)
        title_is_uni = any(kw in title_text.lower() for kw in uni_keywords)

        if title_is_uni and not org_is_uni:
            # Fields are swapped — title holds the university name
            institution = title_text
            degree = org_text
        else:
            institution = org_text
            degree = title_text

        # Abbreviate excessively long college names
        institution = institution.replace(
            "College of Literature, Science, and the Arts",
            "College of LSA",
        )
        degree = degree.replace(
            "College of Literature, Science, and the Arts",
            "College of LSA",
        )

        # Line 1: University name (left, bold) | Dates (right, italic)
        org_para = Paragraph(
            f"<b>{self._esc(institution)}</b>",
            self.styles["entry_left"],
        )
        dates_para = Paragraph(
            self._esc(entry.dates) if entry.dates else "",
            self.styles["entry_dates"],
        )
        tbl = self._make_entry_row(org_para, dates_para, space_before=4)
        elements.append(tbl)

        # Line 2: Degree / title in italics
        if degree:
            elements.append(
                Paragraph(
                    _sanitize_text(self._esc(degree)),
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
            text = " ".join(text.split())
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
        """Create a bullet paragraph with deterministic wrap-aware orphan handling."""
        # Strip <br> tags, literal \n strings, and all extra whitespace
        clean = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
        clean = clean.replace('\\n', ' ').replace('\n', ' ')
        clean = " ".join(clean.split())
        clean = _sanitize_text(clean)

        style = self.styles["bullet"]
        paragraph = self._build_bullet_paragraph(clean, style)
        line_fills = self._measure_paragraph_line_fills(paragraph, style)

        # If the second line is just a tiny orphan, only accept a shrink that
        # truly collapses the bullet back to one line. Otherwise keep wording
        # intact and let upstream selection prefer denser factual bullets.
        if len(line_fills) == 2 and line_fills[1] < 0.30:
            for step in range(1, 3):
                shrink = 0.25 * step
                shrunk = ParagraphStyle(
                    f"BulletShrunk{step}",
                    parent=style,
                    fontSize=style.fontSize - shrink,
                    leading=style.leading - shrink,
                )
                candidate = self._build_bullet_paragraph(clean, shrunk)
                candidate_fills = self._measure_paragraph_line_fills(candidate, shrunk)
                if len(candidate_fills) == 1:
                    return candidate

        return paragraph

    def _build_bullet_paragraph(self, text: str, style: ParagraphStyle) -> Paragraph:
        markup = (
            f'<font color="{COLOR_BULLET.hexval()}">\u2022</font>'
            f"  {self._esc(text)}"
        )
        return Paragraph(markup, style)

    def _measure_paragraph_line_fills(
        self,
        paragraph: Paragraph,
        style: ParagraphStyle,
    ) -> list[float]:
        """Return per-line fill ratios from ReportLab's actual wrapped layout."""
        paragraph.wrap(CONTENT_WIDTH, AVAILABLE_HEIGHT)
        lines = getattr(getattr(paragraph, "blPara", None), "lines", None) or []
        if not lines:
            return []

        first_line_width = max(CONTENT_WIDTH - style.leftIndent - style.firstLineIndent, 1)
        later_line_width = max(CONTENT_WIDTH - style.leftIndent, 1)

        fills: list[float] = []
        for idx, line in enumerate(lines):
            extra_space = self._extract_line_extra_space(line)
            if extra_space is None:
                continue
            available = first_line_width if idx == 0 else later_line_width
            fill_ratio = 1.0 - (extra_space / available)
            fills.append(max(0.0, min(1.0, fill_ratio)))

        return fills

    @staticmethod
    def _extract_line_extra_space(line) -> float | None:
        if isinstance(line, tuple) and line and isinstance(line[0], (int, float)):
            return float(line[0])
        if hasattr(line, "extraSpace") and isinstance(line.extraSpace, (int, float)):
            return float(line.extraSpace)
        return None

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
