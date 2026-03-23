"""Formatter Agent — renders the approved draft as a polished PDF resume.

This agent is purely programmatic (no LLM call). It uses ReportLab to
generate a professionally styled PDF from the structured PipelineState data.
"""

from __future__ import annotations

import logging
import re
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
COLOR_HEADING = HexColor("#1A1A1A")
COLOR_BODY = HexColor("#2D2D2D")
COLOR_SUBTLE = HexColor("#555555")
COLOR_LINK = HexColor("#1A5276")
COLOR_RULE = HexColor("#3B3B3B")

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN_LEFT = 0.6 * inch
MARGIN_RIGHT = 0.6 * inch
MARGIN_TOP = 0.5 * inch
MARGIN_BOTTOM = 0.5 * inch
CONTENT_WIDTH = PAGE_WIDTH - MARGIN_LEFT - MARGIN_RIGHT


# ── Reusable styles ─────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    """Build all paragraph styles used in the resume."""
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "ResumeName",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            textColor=COLOR_HEADING,
            spaceAfter=2,
        ),
        "contact": ParagraphStyle(
            "ResumeContact",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=COLOR_SUBTLE,
            spaceAfter=6,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=11,
            leading=14,
            textColor=COLOR_HEADING,
            spaceBefore=8,
            spaceAfter=2,
        ),
        "entry_title": ParagraphStyle(
            "EntryTitle",
            parent=base["Normal"],
            fontName=FONT_BOLD,
            fontSize=10,
            leading=13,
            textColor=COLOR_HEADING,
            spaceBefore=5,
            spaceAfter=1,
        ),
        "entry_org": ParagraphStyle(
            "EntryOrg",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=10,
            leading=13,
            textColor=COLOR_BODY,
        ),
        "entry_dates": ParagraphStyle(
            "EntryDates",
            parent=base["Normal"],
            fontName=FONT_ITALIC,
            fontSize=10,
            leading=13,
            textColor=COLOR_SUBTLE,
            alignment=TA_RIGHT,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=12.5,
            textColor=COLOR_BODY,
            leftIndent=12,
            spaceAfter=1.5,
        ),
        "skills_category": ParagraphStyle(
            "SkillsCategory",
            parent=base["Normal"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=12.5,
            textColor=COLOR_BODY,
            leftIndent=8,
            spaceAfter=1.5,
        ),
    }


class FormatterAgent:
    """Renders pruned_sections + contact info into a polished PDF file."""

    def __init__(self) -> None:
        self.styles = _build_styles()

    def run(self, state: PipelineState, output_path: Path) -> PipelineState:
        """Build and save the PDF resume."""
        logger.info("▶ Running formatter agent [reportlab — no LLM call]")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            leftMargin=MARGIN_LEFT,
            rightMargin=MARGIN_RIGHT,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
        )

        story: list = []

        # ── Contact header ───────────────────────────────────────
        name, contact_line = self._extract_contact(state.master_resume)
        story.append(Paragraph(self._escape(name), self.styles["name"]))
        if contact_line:
            story.append(Paragraph(contact_line, self.styles["contact"]))

        # ── Resume sections ──────────────────────────────────────
        if state.pruned_sections:
            for section in state.pruned_sections:
                story.extend(self._build_section(section))

        # ── Render PDF ───────────────────────────────────────────
        doc.build(story)
        logger.info("✔ formatter agent complete — saved to %s", output_path)

        return state.model_copy(update={"final_resume": str(output_path)})

    # ── Contact extraction ───────────────────────────────────────

    @staticmethod
    def _extract_contact(master_resume: str) -> tuple[str, str]:
        """Pull name and contact line from the master resume header.

        Returns (name, contact_html) where contact_html contains
        clickable <a href> links for any markdown links found.
        """
        name = "Resume"
        contact_line = ""
        if not master_resume:
            return name, contact_line

        for line in master_resume.split("\n"):
            stripped = line.strip()
            if stripped.startswith("---"):
                break
            # First heading → name
            if stripped.startswith("#") and name == "Resume":
                name = stripped.lstrip("#").strip()
                # Remove markdown bold/italic
                name = re.sub(r"[*_]", "", name).strip()
                # Strip trailing descriptors: "— Master Resume", "- Resume", etc.
                name = re.sub(
                    r"\s*[—–\-]\s*(Master\s+)?Resume.*$", "", name, flags=re.IGNORECASE
                ).strip()
            # Line with | or @ → contact info
            elif ("|" in stripped or "@" in stripped) and name != "Resume":
                contact_line = re.sub(r"\*\*|\*|__|_", "", stripped)
                # Convert markdown links [text](url) → ReportLab <a> tags
                # Skip placeholder URLs like (#) or empty hrefs
                def _link_replacer(m: re.Match) -> str:
                    text, url = m.group(1), m.group(2)
                    if url and url != "#" and url.startswith(("http://", "https://")):
                        return f'<a href="{url}" color="#1A5276">{text}</a>'
                    return text  # Render as plain text if URL is invalid

                contact_line = re.sub(
                    r"\[([^\]]+)\]\(([^)]*)\)",
                    _link_replacer,
                    contact_line,
                )
                contact_line = contact_line.strip()

        return name, contact_line

    # ── Section builders ─────────────────────────────────────────

    def _build_section(self, section: ResumeSection) -> list:
        """Build flowables for one resume section."""
        elements: list = []
        heading_text = section.heading.upper()

        # Section heading
        elements.append(
            Paragraph(self._escape(heading_text), self.styles["section_heading"])
        )
        # Thin rule under heading
        elements.append(
            HRFlowable(
                width="100%",
                thickness=0.5,
                color=COLOR_RULE,
                spaceAfter=4,
                spaceBefore=0,
            )
        )

        is_skills = "skill" in section.heading.lower()

        for entry in section.entries:
            if is_skills:
                elements.extend(self._build_skills_entry(entry))
            else:
                elements.extend(self._build_standard_entry(entry))

        return elements

    def _build_standard_entry(self, entry) -> list:
        """Build a standard entry: title/org left, dates flush-right."""
        elements: list = []

        # ── Title | Org (left) and Dates (right) on the same line ──
        left_parts = []
        left_parts.append(f"<b>{self._escape(entry.title)}</b>")
        if entry.organization:
            left_parts.append(
                f'<font color="#555555">  |  </font>{self._escape(entry.organization)}'
            )

        left_para = Paragraph("".join(left_parts), self.styles["entry_title"])
        right_para = Paragraph(
            self._escape(entry.dates) if entry.dates else "",
            self.styles["entry_dates"],
        )

        # Two-column table: left content expands, dates hug right margin
        col_widths = [CONTENT_WIDTH * 0.75, CONTENT_WIDTH * 0.25]
        row = [[left_para, right_para]]
        tbl = Table(row, colWidths=col_widths)
        tbl.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        elements.append(tbl)

        # ── Bullets ──────────────────────────────────────────────
        for bullet in entry.bullets:
            text = f'<font color="#888888">\u2022</font>  {self._escape(bullet)}'
            elements.append(Paragraph(text, self.styles["bullet"]))

        return elements

    def _build_skills_entry(self, entry) -> list:
        """Build a skills entry — category: items format."""
        elements: list = []
        for bullet in entry.bullets:
            if ":" in bullet:
                category, items = bullet.split(":", 1)
                text = f"<b>{self._escape(category.strip())}:</b> {self._escape(items.strip())}"
            else:
                text = self._escape(bullet)
            elements.append(Paragraph(text, self.styles["skills_category"]))
        return elements

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _escape(text: str) -> str:
        """Escape XML special characters for ReportLab Paragraph markup."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
