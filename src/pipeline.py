"""Pipeline orchestrator — runs agents in sequence with render-and-refine loop + HITL gate."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from pathlib import Path

from .agents import (
    AnalyzerAgent,
    CriticAgent,
    FormatterAgent,
    MapperAgent,
    PrunerAgent,
)
from .state import Evaluation, PipelineState, ResumeSection
from .utils.resume_normalizer import normalize_project_sections
from .utils.resume_parser import parse_source_projects

logger = logging.getLogger(__name__)

# ── Regex for extracting numbers (handles negatives, decimals, comma groups) ──
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_TARGET_PAGE_FILL_RATIO = 0.95
_UNDERFILL_THRESHOLD = 0.80
_MAX_DETERMINISTIC_PRUNE_STEPS = 24
_MAX_EXPAND_STEPS = 12
_MIN_WORK_BULLETS = 3
_MIN_EDUCATION_BULLETS = 2
_MIN_PROJECT_BULLETS = 1


@dataclass(frozen=True)
class _PruneCandidate:
    """A single removable unit considered by the deterministic prune loop.

    The sort key is (relevance_score, section_priority) where lower values
    are removed first.  This means the least-relevant content gets cut
    before more-relevant content, regardless of which section it lives in.
    ``section_priority`` is only a tiebreaker when two candidates share the
    same relevance score.
    """

    kind: str
    section_index: int
    entry_index: int
    bullet_index: int | None
    relevance: float          # the entry's relevance_score (0.0–1.0)
    section_priority: float   # tiebreaker: skills=0, projects=1, education=2, work=3
    label: str


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric tokens from text, normalized (commas stripped)."""
    raw = _NUMBER_RE.findall(text)
    return {n.replace(",", "") for n in raw}


def _sections_to_text(sections) -> str:
    """Serialize all resume section content to plain text for number extraction."""
    parts: list[str] = []
    for section in sections:
        parts.append(section.heading)
        for entry in section.entries:
            parts.append(entry.title)
            parts.append(entry.organization)
            parts.append(entry.dates)
            parts.extend(entry.bullets)
    return "\n".join(parts)


class Pipeline:
    """Orchestrates the 5-agent resume tailoring pipeline.

    Flow:
      Analyzer → Mapper → Pruner → Critic
        → Render-and-Refine loop (test render → overflow? → local prune)
        → HITL gate (1-page confirmed)
        → Formatter saves final PDF
    """

    def __init__(self) -> None:
        self.analyzer = AnalyzerAgent()
        self.mapper = MapperAgent()
        self.pruner = PrunerAgent()
        self.critic = CriticAgent()
        self.formatter = FormatterAgent()

    def run(
        self,
        master_resume: str,
        job_description: str,
        output_path: Path | None = None,
    ) -> PipelineState:
        """Execute the full pipeline."""

        state = PipelineState(
            master_resume=master_resume,
            job_description=job_description,
            source_projects=parse_source_projects(master_resume),
        )

        # ── Step 1: Analyze the JD ───────────────────────────────
        state = self.analyzer.run(state)
        self._print_analysis_summary(state)

        # ── Steps 2–4: Mapper → Pruner → Critic ─────────────────
        state = self._run_core_loop(state)

        # ── Step 5: Render-and-Refine loop ───────────────────────
        state = self._render_and_refine(state)

        if state.last_render_page_count and state.last_render_page_count > 1:
            print(
                f"\n✖  Draft still renders to {state.last_render_page_count} pages "
                "after the refine loop. Refusing to continue to review/save."
            )
            return state

        # ── HITL Gate: pause for human review (1-page confirmed) ─
        action = self._human_review_gate(state)

        while action == "revise":
            if state.revision_count >= state.max_revisions:
                print(
                    f"\n⚠  Max revisions ({state.max_revisions}) reached. "
                    "Proceeding with best draft."
                )
                break
            state = state.model_copy(
                update={
                    "revision_count": state.revision_count + 1,
                    "overflow_pages": None,
                    "last_render_fill_ratio": None,
                    "render_iteration": 0,
                }
            )
            state = self._run_core_loop(state)
            state = self._render_and_refine(state)
            if state.last_render_page_count and state.last_render_page_count > 1:
                print(
                    f"\n✖  Draft still renders to {state.last_render_page_count} pages "
                    "after the revise loop. Refusing to continue to review/save."
                )
                return state
            action = self._human_review_gate(state)

        if action == "reject":
            print("\n✖  Pipeline aborted by user.")
            return state

        # ── Step 6: Save final PDF ───────────────────────────────
        if output_path:
            state = self.formatter.run(state, output_path)
            print(f"\n✔  Resume written to {output_path}")
        else:
            print("\n⚠  No output path specified — skipping PDF generation.")

        return state

    # ── Internal helpers ─────────────────────────────────────────

    def _run_core_loop(self, state: PipelineState) -> PipelineState:
        """Run Mapper → Pruner → [Number Parity Check] → Critic.

        The number parity check is a deterministic regex gate that catches
        fabricated/altered numbers BEFORE the expensive Critic LLM call.
        If it fails, the Mapper is re-invoked with feedback (max 2 retries).

        A soft bullet-floor check catches work-experience entries with <3
        bullets and retries the Mapper with feedback (no crash).
        """
        max_parity_retries = 2

        for attempt in range(max_parity_retries + 1):
            state = self.mapper.run(state)

            # ── Soft bullet-floor check: retry mapper if <3 bullets ──
            sparse_entries = self._find_sparse_entries(state.mapped_sections)
            if sparse_entries and attempt < max_parity_retries:
                logger.warning(
                    "  ⚠ Bullet floor: %s — retrying mapper",
                    ", ".join(f"'{e}'" for e in sparse_entries),
                )
                synthetic_eval = Evaluation(
                    approved=False,
                    factual_drift_issues=[],
                    missing_keywords=[],
                    suggestions=[
                        f"CRITICAL: The following work experience entries have too few "
                        f"bullets (minimum 3 required): {', '.join(sparse_entries)}. "
                        f"Add more specific, detailed accomplishments from the master resume."
                    ],
                    overall_score=0.0,
                )
                state = state.model_copy(update={"evaluation": synthetic_eval})
                continue
            elif sparse_entries:
                logger.warning("  ⚠ Bullet floor still failing after retries — proceeding")

            state = state.model_copy(
                update={
                    "mapped_sections": normalize_project_sections(
                        state.mapped_sections,
                        state.source_projects,
                    )
                }
            )
            state = self.pruner.run(state)
            state = state.model_copy(
                update={
                    "pruned_sections": normalize_project_sections(
                        state.pruned_sections,
                        state.source_projects,
                        max_bullets_per_project=3,
                    )
                }
            )

            # ── Underfill guard: reject pruner output that lost content ──
            state = self._guard_pruner_underfill(state)

            # ── Programmatic number parity gate ──────────────────
            drift_issues = self._check_number_parity(state)

            if not drift_issues:
                logger.info("  ✔ Number parity check passed")
                break  # Clean — proceed to Critic

            # Drift detected — log, inject feedback, retry Mapper
            logger.warning(
                "  ✖ Number parity check FAILED (attempt %d/%d): %s",
                attempt + 1,
                max_parity_retries + 1,
                drift_issues,
            )
            print(f"\n  ✖  Number parity check failed (attempt {attempt + 1}):")
            for issue in drift_issues:
                print(f"     • {issue}")

            if attempt < max_parity_retries:
                # Inject synthetic evaluation so Mapper sees the feedback
                synthetic_eval = Evaluation(
                    approved=False,
                    factual_drift_issues=drift_issues,
                    missing_keywords=[],
                    suggestions=[
                        "CRITICAL: Restore ALL original numbers exactly as they "
                        "appear in the Master Resume. Do not combine, round, or "
                        "alter any numerical values."
                    ],
                    overall_score=0.0,
                )
                state = state.model_copy(update={"evaluation": synthetic_eval})
                print("     → Re-running Mapper with correction feedback...")
            else:
                print("     → Max retries exhausted. Proceeding to Critic.")

        # ── Run Critic on the (hopefully clean) draft ────────────
        state = self.critic.run(state)
        return state

    # Per-section minimum bullet counts (enforced programmatically)
    BULLET_MINIMUMS = {
        "work": 3,       # per entry
        "education": 3,  # per entry
    }

    @classmethod
    def _find_sparse_entries(cls, sections) -> list[str]:
        """Return descriptions of entries that fall below bullet minimums."""
        sparse = []
        if not sections:
            return sparse
        for section in sections:
            heading = section.heading.lower()
            if "experience" in heading or "work" in heading:
                min_bullets = cls.BULLET_MINIMUMS["work"]
                section_label = "Work Experience"
            elif "education" in heading:
                min_bullets = cls.BULLET_MINIMUMS["education"]
                section_label = "Education"
            else:
                continue
            for entry in section.entries:
                if len(entry.bullets) < min_bullets:
                    sparse.append(
                        f"{section_label}: '{entry.title.strip()}' has {len(entry.bullets)} "
                        f"bullets (minimum {min_bullets})"
                    )
        return sparse

    @staticmethod
    def _count_bullets(sections) -> int:
        """Total bullet count across all sections."""
        if not sections:
            return 0
        return sum(len(e.bullets) for s in sections for e in s.entries)

    @staticmethod
    def _count_entries(sections) -> int:
        """Total entry count across all sections."""
        if not sections:
            return 0
        return sum(len(s.entries) for s in sections)

    def _guard_pruner_underfill(self, state: PipelineState) -> PipelineState:
        """If the LLM Pruner dropped bullets or entries, fall back to mapped_sections.

        The pruner's job is quality polish, not content removal.  If it
        removed more than 10% of the bullets or dropped any entries, its
        output is rejected and we use the mapped_sections directly (with
        project normalization applied).
        """
        if not state.mapped_sections or not state.pruned_sections:
            return state

        mapped_bullets = self._count_bullets(state.mapped_sections)
        pruned_bullets = self._count_bullets(state.pruned_sections)
        mapped_entries = self._count_entries(state.mapped_sections)
        pruned_entries = self._count_entries(state.pruned_sections)

        # Allow up to 10% bullet loss (from true redundancy merges)
        bullet_loss = (mapped_bullets - pruned_bullets) / max(mapped_bullets, 1)
        entry_loss = mapped_entries - pruned_entries

        if bullet_loss > 0.10 or entry_loss > 0:
            logger.warning(
                "  ⚠ Pruner over-cut: %d→%d bullets (%.0f%% loss), %d→%d entries. "
                "Falling back to mapped_sections.",
                mapped_bullets, pruned_bullets, bullet_loss * 100,
                mapped_entries, pruned_entries,
            )
            print(
                f"\n  ⚠  Pruner dropped content ({mapped_bullets}→{pruned_bullets} bullets, "
                f"{mapped_entries}→{pruned_entries} entries). Using Mapper output instead."
            )
            state = state.model_copy(
                update={
                    "pruned_sections": normalize_project_sections(
                        state.mapped_sections,
                        state.source_projects,
                        max_bullets_per_project=3,
                    )
                }
            )

        return state

    @staticmethod
    def _check_number_parity(state: PipelineState) -> list[str]:
        """Compare numbers in pruned_sections against master_resume.

        Returns a list of drift issue descriptions (empty = clean).
        """
        if not state.pruned_sections:
            return []

        source_numbers = _extract_numbers(state.master_resume)
        draft_text = _sections_to_text(state.pruned_sections)
        draft_numbers = _extract_numbers(draft_text)

        # Numbers in draft that don't exist anywhere in the source
        phantom_numbers = draft_numbers - source_numbers

        # Filter out trivially safe numbers (single digits 0-9 are too
        # common in dates, list counts, etc. to be meaningful signals)
        phantom_numbers = {n for n in phantom_numbers if len(n) > 1 or int(n) < 0}

        issues: list[str] = []
        for num in sorted(phantom_numbers):
            # Try to find the closest source number for a helpful message
            issues.append(
                f"Number '{num}' found in draft but does not exist in the "
                f"Master Resume. Source numbers include: "
                f"{sorted(source_numbers)[:10]}"
            )

        return issues

    def _render_and_refine(self, state: PipelineState) -> PipelineState:
        """Bidirectional page fitting: prune overflow OR expand underfill.

        1. Measure the current draft.
        2. If >1 page → deterministic prune (remove lowest-relevance units).
        3. If 1 page but <80% fill → deterministic expand (restore from mapped_sections).
        4. Accept when 1 page and fill is in the target band.
        """
        logger.info("▶ Starting render-and-refine loop")
        state = self._enforce_project_retention(state)
        metrics = self.formatter.measure_render(state)
        state = state.model_copy(
            update={
                "last_render_page_count": metrics.page_count,
                "last_render_fill_ratio": metrics.fill_ratio,
                "overflow_pages": None,
                "render_iteration": 0,
            }
        )

        # ── Overflow path ─────────────────────────────────────────
        if metrics.page_count > 1:
            logger.warning(
                "  ⚠ Overflow detected: %d pages (fill %.0f%%) — pruning locally",
                metrics.page_count,
                metrics.fill_ratio * 100,
            )
            print(
                f"\n  ⚠  Render overflow: {metrics.page_count} pages "
                f"({metrics.fill_ratio * 100:.0f}% fill) — pruning locally"
            )

            state = self._prune_to_fit_deterministically(state, metrics)
            state = self._enforce_project_retention(state)

        # ── Underfill path ────────────────────────────────────────
        underfill_metrics = self.formatter.measure_render(state)
        if (
            underfill_metrics.page_count <= 1
            and underfill_metrics.fill_ratio < _UNDERFILL_THRESHOLD
            and state.mapped_sections
        ):
            logger.info(
                "  ⚠ Underfill detected: %.0f%% fill — expanding from mapped pool",
                underfill_metrics.fill_ratio * 100,
            )
            print(
                f"\n  ⚠  Page underfilled ({underfill_metrics.fill_ratio * 100:.0f}% fill) "
                f"— restoring content from Mapper output"
            )
            state = self._expand_to_fill(state, underfill_metrics)

        # ── Final measurement ─────────────────────────────────────
        final_metrics = self.formatter.measure_render(state)
        state = state.model_copy(
            update={
                "overflow_pages": None,
                "render_iteration": 0,
                "last_render_page_count": final_metrics.page_count,
                "last_render_fill_ratio": final_metrics.fill_ratio,
            }
        )
        if final_metrics.page_count > 1:
            print(
                f"\n  ⚠  Could not fit to 1 page after deterministic "
                f"prune steps ({final_metrics.page_count} pages)."
            )
        else:
            logger.info(
                "  ✔ Final render: 1 page, %.0f%% fill",
                final_metrics.fill_ratio * 100,
            )
        return state

    def _enforce_content_floor(self, state: PipelineState, fill_ratio: float) -> PipelineState:
        """Re-run mapper with guidance to add more content when page is underfilled."""
        if not hasattr(self, '_content_floor_attempts'):
            self._content_floor_attempts = 0

        self._content_floor_attempts += 1
        if self._content_floor_attempts > self.MAX_CONTENT_FLOOR_RETRIES:
            logger.warning("  ⚠ Content floor: max retries exhausted (%.0f%% fill) — proceeding", fill_ratio * 100)
            return state

        logger.warning(
            "  ⚠ Content floor: page only %.0f%% filled (minimum %d%%) — "
            "re-running mapper to add content (attempt %d/%d)",
            fill_ratio * 100, int(self.CONTENT_FLOOR_RATIO * 100),
            self._content_floor_attempts, self.MAX_CONTENT_FLOOR_RETRIES,
        )

        # Inject synthetic evaluation telling mapper to add more content
        synthetic_eval = Evaluation(
            approved=False,
            factual_drift_issues=[],
            missing_keywords=state.evaluation.missing_keywords if state.evaluation else [],
            suggestions=[
                f"CRITICAL: The resume is only {fill_ratio:.0%} filled — it MUST fill at least "
                f"{self.CONTENT_FLOOR_RATIO:.0%}. Your CURRENT DRAFT is included below. You must "
                f"keep ALL existing content and ADD MORE. Do NOT remove or shorten any existing "
                f"bullets. Strategies: (1) Add a 4th bullet to work entries that only have 3. "
                f"(2) Expand bullets with specific metrics, tools, and outcomes from the master resume. "
                f"(3) Add more education bullets (honors, relevant coursework, activities). "
                f"(4) Lengthen short one-line bullets into detailed two-liners with context."
            ],
            overall_score=0.3,
        )
        state = state.model_copy(update={"evaluation": synthetic_eval})

        # Re-run mapper → pruner with the feedback
        state = self.mapper.run(state)
        state = state.model_copy(
            update={
                "mapped_sections": normalize_project_sections(
                    state.mapped_sections, state.source_projects,
                )
            }
        )
        state = self.pruner.run(state)
        state = state.model_copy(
            update={
                "pruned_sections": normalize_project_sections(
                    state.pruned_sections, state.source_projects, max_bullets_per_project=3,
                )
            }
        )
        return state

    def _enforce_project_retention(self, state: PipelineState) -> PipelineState:
        """Reject pruned drafts that drop source projects before rendering."""
        if not state.source_projects or not state.pruned_sections:
            return state

        max_restore_attempts = 2
        project_floor_error = (
            "CRITICAL ERROR: You deleted a project. You must include all "
            "projects from the source."
        )

        for attempt in range(1, max_restore_attempts + 1):
            draft_projects = self._get_project_entries(state.pruned_sections)
            missing_projects = self._find_missing_projects(
                state.source_projects,
                draft_projects,
            )
            if not missing_projects:
                return state

            missing_labels = ", ".join(
                self._format_entry_label(entry) for entry in missing_projects
            )
            logger.warning(
                "  ✖ Project retention guardrail failed (attempt %d/%d): %s",
                attempt,
                max_restore_attempts,
                missing_labels,
            )
            print(
                "\n  ✖  Project retention guardrail failed: "
                f"{missing_labels or 'missing source project(s)'}"
            )
            print("     → Rejecting draft and re-running Pruner with source inventory...")

            feedback = [project_floor_error]
            if missing_labels:
                feedback.append(f"Missing project(s): {missing_labels}")

            state = state.model_copy(
                update={
                    "pruner_feedback": feedback,
                    "force_source_project_inventory": True,
                }
            )
            state = self.pruner.run(state)
            state = state.model_copy(
                update={
                    "pruned_sections": normalize_project_sections(
                        state.pruned_sections,
                        state.source_projects,
                        max_bullets_per_project=3,
                    ),
                    "pruner_feedback": [],
                    "force_source_project_inventory": False,
                }
            )

        missing_projects = self._find_missing_projects(
            state.source_projects,
            self._get_project_entries(state.pruned_sections),
        )
        if missing_projects:
            logger.warning(
                "  ⚠ Project retention guardrail fell back to deterministic restore"
            )
            print(
                "     → Pruner still dropped projects; restoring missing source "
                "projects deterministically before render."
            )
            state = state.model_copy(
                update={
                    "pruned_sections": self._restore_missing_projects(
                        state.pruned_sections,
                        state.source_projects,
                    )
                }
            )

        return state

    def _prune_to_fit_deterministically(
        self,
        state: PipelineState,
        metrics,
    ) -> PipelineState:
        """Trim overflow locally using real render metrics instead of more LLM calls."""
        current_state = state
        current_metrics = metrics

        for step in range(1, _MAX_DETERMINISTIC_PRUNE_STEPS + 1):
            if current_metrics.page_count <= 1:
                return current_state

            candidates = self._enumerate_prune_candidates(current_state.pruned_sections)
            if not candidates:
                logger.warning("  ⚠ Deterministic prune stopped: no removable units remain")
                break

            options = []

            for candidate in candidates:
                candidate_sections = self._apply_prune_candidate(
                    current_state.pruned_sections,
                    candidate,
                )
                candidate_state = current_state.model_copy(
                    update={"pruned_sections": candidate_sections}
                )
                candidate_metrics = self.formatter.measure_render(candidate_state)
                if not self._candidate_improves(current_metrics, candidate_metrics):
                    continue
                options.append((candidate, candidate_sections, candidate_metrics))

            if not options:
                logger.warning("  ⚠ Deterministic prune stopped: no candidate improved overflow")
                break

            best_candidate, best_sections, best_metrics = self._select_best_candidate(
                options
            )

            current_state = current_state.model_copy(
                update={
                    "pruned_sections": best_sections,
                    "last_render_page_count": best_metrics.page_count,
                    "last_render_fill_ratio": best_metrics.fill_ratio,
                    "render_iteration": step,
                    "overflow_pages": None,
                }
            )
            current_metrics = best_metrics

            logger.info(
                "  ✔ Deterministic prune step %d: %s -> %d page(s), fill %.0f%%",
                step,
                best_candidate.label,
                best_metrics.page_count,
                best_metrics.fill_ratio * 100,
            )
            print(
                f"     → Local prune {step}: {best_candidate.label} "
                f"→ {best_metrics.page_count} page(s), {best_metrics.fill_ratio * 100:.0f}% fill"
            )

        return current_state

    def _expand_to_fill(self, state: PipelineState, metrics) -> PipelineState:
        """Restore bullets/entries from mapped_sections to fill an underfilled page.

        Walks through mapped_sections looking for content that exists in the
        Mapper output but was removed from pruned_sections (by the LLM Pruner
        or the deterministic pruner).  Adds the highest-relevance missing
        content one unit at a time, re-rendering after each addition, and
        stops when the fill ratio is in the target band or adding more would
        overflow.
        """
        current_state = state
        current_metrics = metrics

        if not state.mapped_sections:
            return state

        for step in range(1, _MAX_EXPAND_STEPS + 1):
            if current_metrics.fill_ratio >= _UNDERFILL_THRESHOLD:
                return current_state

            # Find restorable content: bullets in mapped but missing from pruned
            candidates = self._enumerate_expand_candidates(
                current_state.pruned_sections,
                state.mapped_sections,
            )
            if not candidates:
                logger.info("  ✔ No more content to restore from Mapper pool")
                break

            # Try the highest-relevance candidate first
            for candidate in candidates:
                expanded_sections = self._apply_expand_candidate(
                    current_state.pruned_sections,
                    candidate,
                )
                expanded_state = current_state.model_copy(
                    update={"pruned_sections": expanded_sections}
                )
                expanded_metrics = self.formatter.measure_render(expanded_state)

                if expanded_metrics.page_count > 1:
                    # Adding this would overflow — skip it, try next
                    continue

                # Accept this expansion
                current_state = expanded_state
                current_metrics = expanded_metrics
                logger.info(
                    "  ✔ Expand step %d: %s → fill %.0f%%",
                    step, candidate["label"], current_metrics.fill_ratio * 100,
                )
                print(
                    f"     → Expand {step}: {candidate['label']} "
                    f"→ {current_metrics.fill_ratio * 100:.0f}% fill"
                )
                break
            else:
                # No candidate could be added without overflow
                logger.info("  ✔ Expansion stopped: all remaining candidates would overflow")
                break

        return current_state

    def _enumerate_expand_candidates(self, pruned_sections, mapped_sections) -> list[dict]:
        """Find bullets present in mapped_sections but missing from pruned_sections.

        Returns a list of dicts sorted by relevance (highest first), each
        describing one restorable unit.
        """
        candidates: list[dict] = []

        # Index pruned content for fast lookup
        pruned_bullets_by_entry: dict[tuple[str, str], set[str]] = {}
        pruned_entry_keys: set[tuple[str, str]] = set()
        for section in (pruned_sections or []):
            for entry in section.entries:
                key = (self._section_kind(section.heading), entry.title.strip().lower())
                pruned_entry_keys.add(key)
                pruned_bullets_by_entry[key] = {b.strip().lower() for b in entry.bullets}

        for section in (mapped_sections or []):
            section_kind = self._section_kind(section.heading)
            for entry in section.entries:
                relevance = entry.relevance_score if entry.relevance_score is not None else 0.5
                key = (section_kind, entry.title.strip().lower())

                if key not in pruned_entry_keys and section_kind != "skills":
                    # Whole entry is missing — can restore it
                    candidates.append({
                        "type": "entry",
                        "section_kind": section_kind,
                        "section_heading": section.heading,
                        "entry": entry,
                        "relevance": relevance,
                        "label": f"restore entry {entry.title}",
                    })
                    continue

                # Check for missing bullets within an existing entry
                existing_bullets = pruned_bullets_by_entry.get(key, set())
                for bullet in entry.bullets:
                    if bullet.strip().lower() not in existing_bullets:
                        candidates.append({
                            "type": "bullet",
                            "section_kind": section_kind,
                            "entry_title": entry.title,
                            "bullet": bullet,
                            "relevance": relevance,
                            "label": f"restore bullet in {entry.title}",
                        })

        # Sort by relevance descending — restore highest-value content first
        candidates.sort(key=lambda c: -c["relevance"])
        return candidates

    def _apply_expand_candidate(self, sections, candidate: dict) -> list[ResumeSection]:
        """Apply one expansion and return a new section list."""
        cloned = [s.model_copy(deep=True) for s in (sections or [])]

        if candidate["type"] == "entry":
            # Find or create the target section
            target_idx = None
            for idx, section in enumerate(cloned):
                if self._section_kind(section.heading) == candidate["section_kind"]:
                    target_idx = idx
                    break

            if target_idx is not None:
                cloned[target_idx].entries.append(candidate["entry"].model_copy(deep=True))
            else:
                cloned.append(
                    ResumeSection(
                        heading=candidate["section_heading"],
                        entries=[candidate["entry"].model_copy(deep=True)],
                    )
                )
            return cloned

        if candidate["type"] == "bullet":
            for section in cloned:
                if self._section_kind(section.heading) != candidate["section_kind"]:
                    continue
                for entry in section.entries:
                    if entry.title.strip().lower() == candidate["entry_title"].strip().lower():
                        entry.bullets.append(candidate["bullet"])
                        return cloned

        return cloned

    def _enumerate_prune_candidates(self, sections) -> list[_PruneCandidate]:
        """Return removable units sorted by relevance (lowest relevance removed first).

        Section minimums are still enforced — we won't enumerate a candidate
        that would violate a floor (e.g. dropping a work role below 3 bullets).
        Beyond that, the *only* ranking signal is the entry's relevance_score
        as set by the Mapper.  Section kind is a tiebreaker, not a gate.

        Section priority tiebreaker (lower = cut first when relevance ties):
            skills = 0, projects = 1, education = 2, work = 3
        """
        _SECTION_PRIORITY = {"skills": 0.0, "projects": 1.0, "education": 2.0, "work": 3.0, "other": 1.5}

        candidates: list[_PruneCandidate] = []
        if not sections:
            return candidates

        for section_index, section in enumerate(sections):
            section_kind = self._section_kind(section.heading)
            priority = _SECTION_PRIORITY.get(section_kind, 1.5)

            for entry_index, entry in enumerate(section.entries):
                bullets = entry.bullets or []
                relevance = entry.relevance_score if entry.relevance_score is not None else 0.5

                if section_kind == "skills":
                    # Skills entries don't carry individual relevance; use 0.0
                    # so they're naturally among the first things cut.
                    for bullet_index in range(len(bullets) - 1, -1, -1):
                        candidates.append(
                            _PruneCandidate(
                                kind="skills_line",
                                section_index=section_index,
                                entry_index=entry_index,
                                bullet_index=bullet_index,
                                relevance=0.0,
                                section_priority=priority,
                                label=f"remove skills line {bullet_index + 1}",
                            )
                        )
                    continue

                # Determine per-section bullet floor
                if section_kind == "education":
                    min_bullets = _MIN_EDUCATION_BULLETS
                elif section_kind == "projects":
                    min_bullets = _MIN_PROJECT_BULLETS
                elif section_kind == "work":
                    min_bullets = _MIN_WORK_BULLETS
                else:
                    min_bullets = 0

                # Individual bullet removals (last bullet first within an entry)
                for bullet_index in range(len(bullets) - 1, min_bullets - 1, -1):
                    candidates.append(
                        _PruneCandidate(
                            kind=f"{section_kind}_bullet",
                            section_index=section_index,
                            entry_index=entry_index,
                            bullet_index=bullet_index,
                            relevance=relevance,
                            section_priority=priority,
                            label=f"trim {section_kind} bullet {bullet_index + 1} from {entry.title}",
                        )
                    )

                # Whole-entry removal (only if section has enough entries)
                if section_kind == "work" and len(section.entries) > 2:
                    candidates.append(
                        _PruneCandidate(
                            kind="work_entry",
                            section_index=section_index,
                            entry_index=entry_index,
                            bullet_index=None,
                            relevance=relevance,
                            section_priority=priority,
                            label=f"remove work entry {entry.title}",
                        )
                    )

        return candidates

    @staticmethod
    def _candidate_improves(current_metrics, candidate_metrics) -> bool:
        """Accept candidates that reduce page count or meaningfully shrink overflow."""
        return (
            candidate_metrics.page_count < current_metrics.page_count
            or candidate_metrics.fill_ratio < (current_metrics.fill_ratio - 1e-6)
        )

    @staticmethod
    def _select_best_candidate(options):
        """Pick the best single removal to make this step.

        If any candidate brings us to 1 page, pick the one closest to
        the target fill ratio (preferring to keep more content).  If no
        candidate fits on 1 page yet, pick the one that cuts the most
        overflow while sacrificing the least relevance.
        """
        fitting = [option for option in options if option[2].page_count <= 1]
        if fitting:
            return min(
                fitting,
                key=lambda option: (
                    abs(option[2].fill_ratio - _TARGET_PAGE_FILL_RATIO),
                    option[0].relevance,  # prefer cutting lower-relevance
                ),
            )

        return min(
            options,
            key=lambda option: (
                option[2].page_count,
                option[2].fill_ratio,
                option[0].relevance,        # cut lowest relevance first
                option[0].section_priority,  # tiebreaker: skills before work
            ),
        )

    def _apply_prune_candidate(self, sections, candidate: _PruneCandidate) -> list[ResumeSection]:
        """Apply one local trim and return a cleaned copy of the sections."""
        cloned_sections = [section.model_copy(deep=True) for section in sections or []]
        section = cloned_sections[candidate.section_index]

        if candidate.kind == "work_entry":
            del section.entries[candidate.entry_index]
            return self._cleanup_sections(cloned_sections)

        entry = section.entries[candidate.entry_index]
        if candidate.bullet_index is not None:
            del entry.bullets[candidate.bullet_index]

        return self._cleanup_sections(cloned_sections)

    def _cleanup_sections(self, sections) -> list[ResumeSection]:
        """Drop empty skills entries/sections after a local prune step."""
        cleaned_sections: list[ResumeSection] = []
        for section in sections or []:
            section_kind = self._section_kind(section.heading)
            cleaned_entries = []
            for entry in section.entries:
                cleaned_bullets = [bullet for bullet in entry.bullets if bullet.strip()]
                if section_kind == "skills" and not cleaned_bullets:
                    continue
                cleaned_entries.append(entry.model_copy(update={"bullets": cleaned_bullets}))

            if cleaned_entries:
                cleaned_sections.append(section.model_copy(update={"entries": cleaned_entries}))

        return cleaned_sections

    @staticmethod
    def _section_kind(heading: str) -> str:
        """Map a human heading to a stable section kind."""
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

    def _find_sparse_work_entries(self, sections, min_bullets: int) -> list[str]:
        """Return work entries that violate the soft bullet floor."""
        violations: list[str] = []
        if not sections:
            return violations

        for section in sections:
            if self._section_kind(section.heading) != "work":
                continue
            for entry in section.entries:
                if len(entry.bullets) < min_bullets:
                    violations.append(
                        f"{entry.title} has {len(entry.bullets)} bullets (minimum {min_bullets})"
                    )

        return violations

    @staticmethod
    def _get_project_entries(sections) -> list:
        """Return all project entries from section objects."""
        if not sections:
            return []

        projects = []
        for section in sections:
            if "project" in section.heading.lower():
                projects.extend(section.entries)
        return projects

    @staticmethod
    def _entry_key(entry) -> tuple[str, str]:
        """Normalize title/org for stable project identity checks."""
        normalize = lambda text: re.sub(r"\s+", " ", text).strip().lower()
        project_name = entry.organization.strip() or entry.title.strip()
        project_name = re.split(r"\s*[—–-]\s*", project_name, maxsplit=1)[0]
        return normalize(entry.title), normalize(project_name)

    def _find_missing_projects(self, source_projects, draft_projects) -> list:
        """Return source project entries missing from the draft."""
        draft_keys = {self._entry_key(entry) for entry in draft_projects}
        return [
            entry
            for entry in source_projects
            if self._entry_key(entry) not in draft_keys
        ]

    @staticmethod
    def _format_entry_label(entry) -> str:
        """Human-readable label for logs and guardrail feedback."""
        parts = [entry.title.strip(), entry.organization.strip()]
        return " | ".join(part for part in parts if part)

    def _restore_missing_projects(self, sections, source_projects):
        """Merge missing source projects back into the Projects section."""
        restored_sections = list(sections or [])
        projects_index = next(
            (
                idx
                for idx, section in enumerate(restored_sections)
                if "project" in section.heading.lower()
            ),
            None,
        )

        if projects_index is None:
            restored_sections.append(
                ResumeSection(heading="Projects", entries=list(source_projects))
            )
            return restored_sections

        draft_projects = restored_sections[projects_index].entries
        draft_by_key = {self._entry_key(entry): entry for entry in draft_projects}
        merged_projects = [
            draft_by_key.get(self._entry_key(entry), entry)
            for entry in source_projects
        ]

        restored_sections[projects_index] = restored_sections[projects_index].model_copy(
            update={"entries": merged_projects}
        )
        return restored_sections

    def _human_review_gate(self, state: PipelineState) -> str:
        """Display the Critic evaluation + draft preview, then block for user input."""
        print("\n" + "=" * 70)
        print("  HUMAN REVIEW GATE — Pipeline paused for your approval")
        page_count = state.last_render_page_count or "unknown"
        print(f"  (Latest test render: {page_count} page(s))")
        print("=" * 70)

        # Show Critic evaluation
        if state.evaluation:
            ev = state.evaluation
            status = "✔ APPROVED" if ev.approved else "✖ NEEDS REVISION"
            print(f"\n  Critic verdict: {status}  (score: {ev.overall_score:.2f})")

            if ev.factual_drift_issues:
                print("\n  ⚠  Factual drift issues:")
                for issue in ev.factual_drift_issues:
                    print(f"     • {issue}")

            if ev.missing_keywords:
                print(f"\n  ⚠  Missing keywords: {', '.join(ev.missing_keywords)}")

            if ev.suggestions:
                print("\n  💡 Suggestions:")
                for s in ev.suggestions:
                    print(f"     • {s}")

        # Show draft preview
        print("\n" + "-" * 70)
        print("  DRAFT PREVIEW (1 page)")
        print("-" * 70)
        if state.pruned_sections:
            for section in state.pruned_sections:
                print(f"\n  ## {section.heading}")
                for entry in section.entries:
                    print(f"  **{entry.title}** | {entry.organization} | {entry.dates}")
                    for bullet in entry.bullets:
                        print(f"    - {bullet}")

        print("\n" + "-" * 70)
        revision_note = ""
        if state.revision_count >= state.max_revisions:
            revision_note = " (max revisions reached — revise unavailable)"

        print(f"\n  Actions:  [a]pprove  |  [r]evise{revision_note}  |  re[j]ect")

        while True:
            choice = input("\n  Your choice: ").strip().lower()
            if choice in ("a", "approve"):
                return "approve"
            elif choice in ("r", "revise"):
                if state.revision_count >= state.max_revisions:
                    print("  ⚠  Max revisions reached. Choose approve or reject.")
                    continue
                return "revise"
            elif choice in ("j", "reject"):
                return "reject"
            else:
                print("  Invalid choice. Enter a, r, or j.")

    def _print_analysis_summary(self, state: PipelineState) -> None:
        """Print a brief summary of the JD analysis."""
        if not state.analysis:
            return
        a = state.analysis
        print("\n" + "-" * 70)
        print(f"  JD Analysis: {a.job_title} @ {a.company_name}")
        print(f"  Hard skills: {', '.join(a.hard_skills[:10])}")
        print(f"  Soft skills: {', '.join(a.soft_skills[:5])}")
        if a.experience_years:
            print(f"  Experience:  {a.experience_years}")
        print(f"  Top priority: {', '.join(a.priority_ranking[:5])}")
        print("-" * 70)
