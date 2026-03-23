"""Deterministic cleanup for LLM-produced resume sections."""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from ..state import ResumeEntry, ResumeSection
from .resume_parser import clean_resume_text, looks_like_date_text

logger = logging.getLogger(__name__)


def normalize_project_sections(
    sections: list[ResumeSection] | None,
    source_projects: list[ResumeEntry],
    *,
    max_bullets_per_project: int | None = None,
) -> list[ResumeSection] | None:
    """Sanitize and canonicalize project sections against the source inventory."""
    if not sections:
        return sections

    normalized_sections: list[ResumeSection] = []
    for section in sections:
        sanitized_entries = [_sanitize_entry(entry) for entry in section.entries]
        if "project" not in section.heading.lower():
            normalized_sections.append(
                section.model_copy(update={"entries": sanitized_entries})
            )
            continue

        normalized_sections.append(
            _canonicalize_project_section(
                section.heading,
                sanitized_entries,
                source_projects,
                max_bullets_per_project=max_bullets_per_project,
            )
        )

    return normalized_sections


def _canonicalize_project_section(
    heading: str,
    entries: list[ResumeEntry],
    source_projects: list[ResumeEntry],
    *,
    max_bullets_per_project: int | None,
) -> ResumeSection:
    """Merge malformed or duplicated project entries onto canonical source projects."""
    buckets: dict[tuple[str, str], dict] = {}
    ordered_keys: list[tuple[str, str]] = []

    for source in source_projects:
        key = _source_key(source)
        ordered_keys.append(key)
        buckets[key] = {
            "source": source,
            "bullets": [],
            "relevance_score": None,
            "matched": False,
        }

    last_matched_key: tuple[str, str] | None = None

    for entry in entries:
        matched_key = _match_source_project(entry, source_projects)
        if matched_key is None and _looks_like_spillover_entry(entry):
            matched_key = last_matched_key

        if matched_key is None:
            logger.warning(
                "Dropping unmatched project entry during normalization: %s | %s",
                entry.title,
                entry.organization,
            )
            continue

        bucket = buckets[matched_key]
        bucket["matched"] = True
        bucket["bullets"].extend(entry.bullets)

        if entry.relevance_score is not None:
            current = bucket["relevance_score"]
            bucket["relevance_score"] = (
                entry.relevance_score
                if current is None
                else max(current, entry.relevance_score)
            )

        last_matched_key = matched_key

    normalized_entries: list[ResumeEntry] = []
    for key in ordered_keys:
        bucket = buckets[key]
        if not bucket["matched"]:
            continue

        source = bucket["source"]
        bullets = _dedupe_preserve_order(bucket["bullets"])
        if max_bullets_per_project is not None:
            bullets = bullets[:max_bullets_per_project]
        if not bullets:
            fallback = [_sanitize_text(b) for b in source.bullets]
            bullets = fallback[: max_bullets_per_project or len(fallback)]

        normalized_entries.append(
            source.model_copy(
                update={
                    "bullets": bullets,
                    "relevance_score": bucket["relevance_score"],
                }
            )
        )

    return ResumeSection(heading=heading, entries=normalized_entries)


def _sanitize_entry(entry: ResumeEntry) -> ResumeEntry:
    """Strip markdown formatting from all user-visible entry fields."""
    return entry.model_copy(
        update={
            "title": _sanitize_text(entry.title),
            "organization": _sanitize_text(entry.organization),
            "dates": _sanitize_text(entry.dates),
            "bullets": [_sanitize_text(bullet) for bullet in entry.bullets],
        }
    )


def _sanitize_text(text: str) -> str:
    return clean_resume_text(text)


def _looks_like_spillover_entry(entry: ResumeEntry) -> bool:
    """Heuristic for malformed entries caused by raw markdown/date spillover."""
    combined = " ".join(part for part in [entry.title, entry.organization] if part)
    lower_combined = combined.lower()

    return bool(
        looks_like_date_text(entry.title)
        or "live:" in lower_combined
        or "http://" in lower_combined
        or "https://" in lower_combined
        or ".vercel.app" in lower_combined
    )


def _match_source_project(
    entry: ResumeEntry,
    source_projects: list[ResumeEntry],
) -> tuple[str, str] | None:
    """Find the best canonical source project for a draft project entry."""
    entry_title = _normalize_key(entry.title)
    entry_org = _normalize_key(_project_name(entry.organization))

    for source in source_projects:
        if entry_title and entry_title == _normalize_key(source.title):
            return _source_key(source)
        if entry_org and entry_org == _normalize_key(_project_name(source.organization)):
            return _source_key(source)

    haystack = " ".join(
        part
        for part in [entry.title, entry.organization, " ".join(entry.bullets[:2])]
        if part
    ).lower()
    best_source: ResumeEntry | None = None
    best_score = 0.0

    for source in source_projects:
        needle = " ".join(
            [
                source.title,
                _project_name(source.organization),
                " ".join(source.bullets[:2]),
            ]
        ).lower()
        score = SequenceMatcher(None, haystack, needle).ratio()
        if score > best_score:
            best_score = score
            best_source = source

    if best_source and best_score >= 0.42:
        return _source_key(best_source)
    return None


def _source_key(entry: ResumeEntry) -> tuple[str, str]:
    return (
        _normalize_key(entry.title),
        _normalize_key(_project_name(entry.organization)),
    )


def _project_name(text: str) -> str:
    return re.split(r"\s*[—–-]\s*", text.strip(), maxsplit=1)[0]


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []

    for item in items:
        cleaned = _sanitize_text(item)
        if not cleaned:
            continue
        key = _normalize_key(cleaned)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)

    return deduped
