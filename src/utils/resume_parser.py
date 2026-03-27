"""Deterministic helpers for parsing resume structure from raw text."""

from __future__ import annotations

import re

from ..state import ResumeEntry, ResumeSection, SourceBullet

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

_STOP_HEADINGS = {
    "master courselist",
    "course list",
    "master course list",
}

_SECTION_ALIASES = {
    "education": "Education",
    "work experience": "Work Experience",
    "experience": "Work Experience",
    "projects": "Projects",
    "skills": "Skills",
    "leadership & activities": "Leadership & Activities",
    "leadership and activities": "Leadership & Activities",
    "activities": "Leadership & Activities",
    "certifications": "Certifications",
}


def build_source_inventory(master_resume: str) -> tuple[list[ResumeSection], list[SourceBullet]]:
    """Parse the full source resume into stable entry + bullet inventories."""
    sections = parse_source_resume(master_resume)
    bullets = build_source_bullet_inventory(sections)
    return sections, bullets


def parse_source_resume(master_resume: str) -> list[ResumeSection]:
    """Parse the full resume into canonical section/entry objects with stable IDs."""
    section_chunks = _split_source_sections(master_resume)
    parsed_sections: list[ResumeSection] = []

    for section_heading, lines in section_chunks:
        if not lines:
            continue

        kind = section_kind(section_heading)
        if kind == "skills":
            section = _parse_skills_section(section_heading, lines)
        elif kind == "other":
            section = _parse_bullet_list_section(section_heading, lines)
        else:
            section = _parse_structured_section(section_heading, lines)

        if section.entries:
            parsed_sections.append(_assign_entry_and_bullet_ids(section))

    return parsed_sections


def build_source_bullet_inventory(source_sections: list[ResumeSection]) -> list[SourceBullet]:
    """Flatten parsed source sections into a bullet-level inventory."""
    bullets: list[SourceBullet] = []
    order = 0

    for section in source_sections:
        kind = section_kind(section.heading)
        for entry in section.entries:
            for bullet_id, bullet_text in zip(entry.bullet_ids, entry.bullets):
                bullets.append(
                    SourceBullet(
                        bullet_id=bullet_id,
                        entry_id=entry.entry_id or "",
                        section_kind=kind,
                        section_heading=section.heading,
                        title=entry.title,
                        organization=entry.organization,
                        dates=entry.dates,
                        text=bullet_text,
                        order=order,
                    )
                )
                order += 1

    return bullets


def parse_source_projects(master_resume: str) -> list[ResumeEntry]:
    """Return canonical source project entries from the full parsed resume."""
    for section in parse_source_resume(master_resume):
        if section_kind(section.heading) == "projects":
            return section.entries
    return []


def extract_source_projects(source_sections: list[ResumeSection]) -> list[ResumeEntry]:
    """Return project entries from already-parsed source sections."""
    for section in source_sections:
        if section_kind(section.heading) == "projects":
            return section.entries
    return []


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
    text = _extract_reportlab_artifact_text(text)
    text = strip_markdown_links(text)
    text = re.sub(r"[*_`]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def section_kind(heading: str) -> str:
    """Map a human section heading to a stable kind string."""
    heading_lower = heading.lower()
    if "skill" in heading_lower:
        return "skills"
    if "edu" in heading_lower:
        return "education"
    if "project" in heading_lower:
        return "projects"
    if "experience" in heading_lower or "work" in heading_lower:
        return "work"
    return "other"


def _split_source_sections(master_resume: str) -> list[tuple[str, list[str]]]:
    lines = [line.rstrip() for line in master_resume.splitlines()]
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()
        normalized = clean_resume_text(stripped)
        if not normalized:
            continue

        if _is_stop_heading(normalized):
            break

        if _is_section_heading(normalized):
            if current_heading and current_lines:
                sections.append((current_heading, current_lines))
            current_heading = _canonicalize_heading(normalized)
            current_lines = []
            continue

        if current_heading:
            current_lines.append(stripped)

    if current_heading and current_lines:
        sections.append((current_heading, current_lines))

    return sections


def _parse_structured_section(section_heading: str, lines: list[str]) -> ResumeSection:
    entries: list[ResumeEntry] = []
    i = 0
    kind = section_kind(section_heading)

    while i < len(lines):
        while i < len(lines) and not clean_resume_text(lines[i]):
            i += 1
        if i >= len(lines):
            break

        if _extract_bullet_text(lines[i]) is not None:
            bullet_entries, i = _consume_bullet_only_entries(
                lines, i, section_heading, start_index=len(entries)
            )
            entries.extend(bullet_entries)
            continue

        header_lines: list[str] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if looks_like_date_text(stripped):
                break
            if _extract_bullet_text(stripped) is not None and not header_lines:
                break
            header_lines.append(stripped)
            i += 1
            if len(header_lines) >= 4 and not _date_within_lookahead(lines, i):
                break

        if not header_lines:
            i += 1
            continue

        dates = ""
        remainder = ""
        if i < len(lines):
            dates, remainder = extract_date_text(lines[i].strip())
            if dates:
                i += 1

        entry = _make_entry_from_header(section_heading, header_lines, dates)
        if remainder:
            if kind == "education" and not entry.title:
                entry.title = clean_resume_text(remainder)
            else:
                entry.bullets.append(clean_resume_text(remainder))

        degree_assigned = bool(entry.title) if kind == "education" else False

        while i < len(lines):
            stripped = lines[i].strip()
            normalized = clean_resume_text(stripped)
            if not normalized:
                i += 1
                continue
            if _looks_like_next_entry(lines, i, kind):
                break

            bullet_text = _extract_bullet_text(stripped)
            if bullet_text is not None:
                cleaned_bullet = clean_resume_text(bullet_text)
                if cleaned_bullet:
                    entry.bullets.append(cleaned_bullet)
                i += 1
                continue

            if kind == "education" and not degree_assigned:
                entry.title = _merge_text(entry.title, normalized) if entry.title else normalized
                degree_assigned = True
            elif not entry.bullets:
                entry.bullets.append(normalized)
            else:
                entry.bullets[-1] = _merge_text(entry.bullets[-1], normalized)
            i += 1

        finalized = _finalize_entry(entry)
        if finalized.title or finalized.organization or finalized.bullets:
            entries.append(finalized)

    return ResumeSection(heading=section_heading, entries=entries)


def _parse_skills_section(section_heading: str, lines: list[str]) -> ResumeSection:
    skill_lines: list[str] = []
    cleaned_lines = [clean_resume_text(line) for line in lines if clean_resume_text(line)]
    i = 0

    while i < len(cleaned_lines):
        line = cleaned_lines[i]
        next_line = cleaned_lines[i + 1] if i + 1 < len(cleaned_lines) else ""

        if ":" in line:
            skill_lines.append(line)
            i += 1
            continue

        if next_line and _looks_like_skill_label(next_line) and _looks_like_skill_items(line):
            skill_lines.append(f"{next_line}: {line}")
            i += 2
            continue

        if next_line and _looks_like_skill_label(line) and _looks_like_skill_items(next_line):
            skill_lines.append(f"{line}: {next_line}")
            i += 2
            continue

        skill_lines.append(line)
        i += 1

    entry = ResumeEntry(
        title="",
        organization="",
        dates="",
        bullets=skill_lines,
    )
    return ResumeSection(heading=section_heading, entries=[_finalize_entry(entry)])


def _parse_bullet_list_section(
    section_heading: str,
    lines: list[str],
) -> ResumeSection:
    entries, _ = _consume_bullet_only_entries(lines, 0, section_heading, start_index=0)
    return ResumeSection(heading=section_heading, entries=entries)


def _consume_bullet_only_entries(
    lines: list[str],
    start: int,
    section_heading: str,
    *,
    start_index: int,
) -> tuple[list[ResumeEntry], int]:
    entries: list[ResumeEntry] = []
    i = start
    bullet_counter = start_index

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue

        bullet_text = _extract_bullet_text(stripped)
        if bullet_text is None:
            break

        text = clean_resume_text(bullet_text)
        i += 1
        while i < len(lines):
            continuation = lines[i].strip()
            if not continuation:
                i += 1
                continue
            if _extract_bullet_text(continuation) is not None:
                break
            if _is_section_heading(clean_resume_text(continuation)) or _is_stop_heading(clean_resume_text(continuation)):
                break
            text = _merge_text(text, continuation)
            i += 1

        title, detail = _split_bullet_title(text)
        entry = ResumeEntry(
            title=title,
            organization="",
            dates="",
            bullets=[detail or text],
        )
        finalized = _finalize_entry(entry)
        if finalized.title or finalized.bullets:
            entries.append(finalized)
            bullet_counter += 1

    return entries, i


def _make_entry_from_header(section_heading: str, header_lines: list[str], dates: str) -> ResumeEntry:
    header_text = clean_resume_text(" ".join(header_lines))
    kind = section_kind(section_heading)

    if kind == "education":
        return ResumeEntry(
            title="",
            organization=header_text,
            dates=clean_resume_text(dates),
            bullets=[],
        )

    maybe_entry = _parse_entry_header(header_text)
    if maybe_entry:
        return maybe_entry.model_copy(update={"dates": clean_resume_text(dates)})

    return ResumeEntry(
        title=header_text,
        organization="",
        dates=clean_resume_text(dates),
        bullets=[],
    )


def _assign_entry_and_bullet_ids(section: ResumeSection) -> ResumeSection:
    kind = section_kind(section.heading)
    assigned_entries: list[ResumeEntry] = []

    for index, entry in enumerate(section.entries, start=1):
        entry_id = _make_entry_id(kind, entry, index)
        bullet_ids = [f"{entry_id}::b{bullet_index}" for bullet_index in range(1, len(entry.bullets) + 1)]
        assigned_entries.append(
            entry.model_copy(
                update={
                    "entry_id": entry_id,
                    "bullet_ids": bullet_ids,
                }
            )
        )

    return section.model_copy(update={"entries": assigned_entries})


def _make_entry_id(kind: str, entry: ResumeEntry, index: int) -> str:
    slug_parts = [entry.title, entry.organization]
    slug = _slug("-".join(part for part in slug_parts if part))
    if not slug:
        slug = f"{kind}-entry"
    return f"{kind}::{slug}::{index}"


def _slug(text: str) -> str:
    text = clean_resume_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")[:80]


def _looks_like_next_entry(lines: list[str], start_index: int, kind: str) -> bool:
    header_lines: list[str] = []
    for idx in range(start_index, min(start_index + 4, len(lines))):
        stripped = lines[idx].strip()
        if not stripped:
            continue
        if _extract_bullet_text(stripped) is not None:
            return False
        if looks_like_date_text(stripped):
            if not header_lines:
                return False
            if kind in {"work", "projects"}:
                return "|" in header_lines[0]
            return True
        header_lines.append(stripped)
    return False


def _date_within_lookahead(lines: list[str], start_index: int) -> bool:
    for idx in range(start_index, min(start_index + 3, len(lines))):
        if looks_like_date_text(lines[idx].strip()):
            return True
    return False


def _is_section_heading(line: str) -> bool:
    normalized = line.lower()
    if _is_stop_heading(normalized):
        return True
    if normalized in _SECTION_ALIASES:
        return True
    if _MARKDOWN_HEADING_RE.match(line):
        return True
    if _UPPER_HEADING_RE.match(line) and len(line) <= 40:
        return True
    return False


def _canonicalize_heading(line: str) -> str:
    normalized = re.sub(r"^##\s*", "", line).strip().lower()
    return _SECTION_ALIASES.get(normalized, clean_resume_text(line).title())


def _is_stop_heading(line: str) -> bool:
    normalized = re.sub(r"^##\s*", "", line).strip().lower()
    return normalized in _STOP_HEADINGS


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
    entry.bullet_ids = entry.bullet_ids[: len(entry.bullets)]
    return entry


def _looks_like_skill_label(text: str) -> bool:
    cleaned = clean_resume_text(text)
    return bool(cleaned) and len(cleaned.split()) <= 4 and "," not in cleaned and ":" not in cleaned


def _looks_like_skill_items(text: str) -> bool:
    cleaned = clean_resume_text(text)
    return bool(cleaned) and ("," in cleaned or "·" in cleaned or len(cleaned.split()) >= 3)


def _split_bullet_title(text: str) -> tuple[str, str]:
    cleaned = clean_resume_text(text)
    if ":" in cleaned:
        title, detail = cleaned.split(":", 1)
        return title.strip(), detail.strip()
    if "—" in cleaned:
        title, detail = cleaned.split("—", 1)
        return title.strip(), detail.strip()
    return cleaned, cleaned


def _extract_reportlab_artifact_text(text: str) -> str:
    match = re.search(r"text':\s*'(.+?)'\s+'frags':", text, flags=re.DOTALL)
    if match:
        text = match.group(1)
    text = re.sub(r"Paragraph\(.+?#Paragraph", "", text, flags=re.DOTALL)
    return text
