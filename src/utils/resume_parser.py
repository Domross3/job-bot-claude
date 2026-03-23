"""Deterministic helpers for parsing resume structure from raw text."""

from __future__ import annotations

import re

from ..state import ResumeEntry

_PROJECTS_HEADING_RE = re.compile(r"^(?:##\s*)?projects\s*$", re.IGNORECASE)
_MARKDOWN_HEADING_RE = re.compile(r"^##\s+.+$")
_UPPER_HEADING_RE = re.compile(r"^[A-Z][A-Z\s/&-]{2,}$")
_MARKDOWN_ENTRY_RE = re.compile(r"^###\s+(?P<title>[^|]+?)\s*\|\s*(?P<org>.+)$")
_PLAIN_ENTRY_RE = re.compile(r"^(?P<title>[^|]+?)\s*\|\s*(?P<org>.+)$")
_BULLET_RE = re.compile(r"^[\-\*\u2022]\s*(.+)$")
_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MONTH_RE = (
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"(?:[a-z]+)?\.?"
)
_DATE_RE = re.compile(
    rf"^(?:{_MONTH_RE}\s+\d{{4}}|\d{{4}})\s*[–—-]\s*"
    rf"(?:Present|(?:{_MONTH_RE}\s+\d{{4}})|\d{{4}})$",
    re.IGNORECASE,
)


def parse_source_projects(master_resume: str) -> list[ResumeEntry]:
    """Parse project entries from a Markdown or PDF-extracted master resume."""
    lines = [line.rstrip() for line in master_resume.splitlines()]
    in_projects = False
    projects: list[ResumeEntry] = []
    current: ResumeEntry | None = None

    for raw_line in lines:
        line = raw_line.strip()
        normalized_line = clean_resume_text(line)
        if not line:
            continue

        if not in_projects:
            if _PROJECTS_HEADING_RE.match(normalized_line):
                in_projects = True
            continue

        if _is_new_top_level_heading(normalized_line):
            break

        maybe_entry = _parse_entry_header(line)
        if maybe_entry:
            if current:
                projects.append(_finalize_entry(current))
            current = maybe_entry
            continue

        if current is None:
            continue

        date_text, _ = extract_date_text(line)
        if not current.dates and date_text:
            current.dates = date_text
            continue

        bullet_text = _extract_bullet_text(line)
        if bullet_text is not None:
            current.bullets.append(bullet_text)
            continue

        if normalized_line.lower().startswith("tech:"):
            current.bullets.append(normalized_line)
            continue

        if current.bullets:
            current.bullets[-1] = _merge_text(current.bullets[-1], normalized_line)
        elif current.dates:
            current.organization = _merge_text(current.organization, normalized_line)
        else:
            current.organization = _merge_text(current.organization, normalized_line)

    if current:
        projects.append(_finalize_entry(current))

    return projects


def _is_new_top_level_heading(line: str) -> bool:
    if _PROJECTS_HEADING_RE.match(line):
        return False
    if _MARKDOWN_HEADING_RE.match(line):
        return True
    if _UPPER_HEADING_RE.match(line) and len(line) <= 40:
        return True
    return False


def _parse_entry_header(line: str) -> ResumeEntry | None:
    normalized_line = clean_resume_text(line)

    markdown_match = _MARKDOWN_ENTRY_RE.match(normalized_line)
    if markdown_match:
        return ResumeEntry(
            title=clean_resume_text(markdown_match.group("title")),
            organization=clean_resume_text(markdown_match.group("org")),
            dates="",
            bullets=[],
        )

    plain_match = _PLAIN_ENTRY_RE.match(normalized_line)
    if plain_match and not looks_like_date_text(plain_match.group("title")):
        return ResumeEntry(
            title=clean_resume_text(plain_match.group("title")),
            organization=clean_resume_text(plain_match.group("org")),
            dates="",
            bullets=[],
        )

    return None


def _extract_bullet_text(line: str) -> str | None:
    match = _BULLET_RE.match(line)
    if not match:
        return None
    return clean_resume_text(match.group(1))


def _merge_text(existing: str, new_text: str) -> str:
    return clean_resume_text(f"{existing} {new_text}")


def _finalize_entry(entry: ResumeEntry) -> ResumeEntry:
    entry.title = clean_resume_text(entry.title)
    entry.organization = clean_resume_text(entry.organization)
    entry.dates = clean_resume_text(entry.dates)
    entry.bullets = [clean_resume_text(b) for b in entry.bullets if clean_resume_text(b)]
    return entry


def extract_date_text(text: str) -> tuple[str, str]:
    """Return (date, remainder) if a line begins with a date range."""
    normalized = clean_resume_text(text)
    if "|" in normalized:
        first, remainder = (part.strip() for part in normalized.split("|", 1))
        if _DATE_RE.match(first):
            return first, remainder
    if _DATE_RE.match(normalized):
        return normalized, ""
    return "", ""


def looks_like_date_text(text: str) -> bool:
    """Whether the text begins with a date range after markdown cleanup."""
    date_text, _ = extract_date_text(text)
    return bool(date_text)


def strip_markdown_links(text: str) -> str:
    """Collapse Markdown links to their visible label."""
    return _MARKDOWN_LINK_RE.sub(r"\1", text)


def clean_resume_text(text: str) -> str:
    """Normalize markdown-heavy resume text into plain text."""
    text = strip_markdown_links(text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
