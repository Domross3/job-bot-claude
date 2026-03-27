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
from .state import Evaluation, PipelineState, ResumeEntry, ResumeSection
from .utils.resume_normalizer import normalize_project_sections
from .utils.resume_parser import (
    build_source_inventory,
    extract_source_projects,
    section_kind,
)

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
    entry_id: str
    bullet_id: str | None
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

    @staticmethod
    def build_initial_state(master_resume: str, job_description: str) -> PipelineState:
        """Construct the initial state with deterministic source inventories."""
        source_sections, source_bullets = build_source_inventory(master_resume)
        return PipelineState(
            master_resume=master_resume,
            job_description=job_description,
            source_sections=source_sections,
            source_bullets=source_bullets,
            source_projects=extract_source_projects(source_sections),
        )

    def run(
        self,
        master_resume: str,
        job_description: str,
        output_path: Path | None = None,
    ) -> PipelineState:
        """Execute the full pipeline."""

        state = self.build_initial_state(master_resume, job_description)

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
            state = self._materialize_mapped_draft(state)

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
            state = self._capture_polished_bullet_texts(state)

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

    def _materialize_mapped_draft(self, state: PipelineState) -> PipelineState:
        """Select an initial draft deterministically from the full mapped bullet pool."""
        selected_bullet_ids = self._select_initial_bullet_ids(state)
        state = state.model_copy(
            update={
                "selected_bullet_ids": selected_bullet_ids,
                "polished_bullet_texts": {},
            }
        )
        mapped_sections = self._assemble_sections_from_selection(state, selected_bullet_ids)
        return state.model_copy(update={"mapped_sections": mapped_sections})

    def _select_initial_bullet_ids(self, state: PipelineState) -> list[str]:
        """Choose an initial dense draft from the full mapped bullet inventory."""
        score_lookup = {bullet.bullet_id: bullet.relevance_score for bullet in state.mapped_bullets}
        selected: list[str] = []

        for section in state.source_sections:
            kind = self._section_kind(section.heading)

            if kind == "work":
                ranked_entries = sorted(
                    section.entries,
                    key=lambda entry: self._entry_relevance(entry, score_lookup),
                    reverse=True,
                )[:4]
                keep_entry_ids = {entry.entry_id for entry in ranked_entries}
                for entry in section.entries:
                    if entry.entry_id not in keep_entry_ids:
                        continue
                    target = 4 if self._entry_relevance(entry, score_lookup) >= 0.80 else 3
                    selected.extend(
                        self._pick_top_bullets(
                            entry,
                            score_lookup,
                            target=min(target, len(entry.bullet_ids)),
                        )
                    )
                continue

            if kind == "education":
                for entry in section.entries:
                    selected.extend(
                        self._pick_top_bullets(
                            entry,
                            score_lookup,
                            target=min(4, len(entry.bullet_ids)),
                        )
                    )
                continue

            if kind == "projects":
                for entry in section.entries:
                    selected.extend(
                        self._pick_top_bullets(
                            entry,
                            score_lookup,
                            target=min(2, len(entry.bullet_ids)),
                        )
                    )
                continue

            if kind == "skills":
                for entry in section.entries:
                    selected.extend(
                        self._pick_top_bullets(
                            entry,
                            score_lookup,
                            target=min(4, len(entry.bullet_ids)),
                        )
                    )
                continue

            ranked_other = sorted(
                section.entries,
                key=lambda entry: self._entry_relevance(entry, score_lookup),
                reverse=True,
            )[:2]
            keep_other_ids = {entry.entry_id for entry in ranked_other}
            for entry in section.entries:
                if entry.entry_id not in keep_other_ids:
                    continue
                selected.extend(
                    self._pick_top_bullets(
                        entry,
                        score_lookup,
                        target=min(1, len(entry.bullet_ids)),
                    )
                )

        return selected

    @staticmethod
    def _pick_top_bullets(entry: ResumeEntry, score_lookup: dict[str, float], target: int) -> list[str]:
        """Pick the strongest bullets for one entry, preserving source order."""
        if target <= 0:
            return []

        ranked = sorted(
            enumerate(entry.bullet_ids),
            key=lambda pair: (score_lookup.get(pair[1], 0.0), -pair[0]),
            reverse=True,
        )[:target]
        keep_ids = {bullet_id for _, bullet_id in ranked}
        return [bullet_id for bullet_id in entry.bullet_ids if bullet_id in keep_ids]

    @staticmethod
    def _entry_relevance(entry: ResumeEntry, score_lookup: dict[str, float]) -> float:
        if not entry.bullet_ids:
            return 0.0
        return max((score_lookup.get(bullet_id, 0.0) for bullet_id in entry.bullet_ids), default=0.0)

    def _assemble_sections_from_selection(
        self,
        state: PipelineState,
        selected_bullet_ids: list[str],
    ) -> list[ResumeSection]:
        """Rebuild ordered resume sections from the deterministic selection set."""
        selected_set = set(selected_bullet_ids)
        score_lookup = {bullet.bullet_id: bullet.relevance_score for bullet in state.mapped_bullets}
        text_lookup = {
            bullet.bullet_id: bullet.rewritten_text
            for bullet in state.mapped_bullets
        }
        text_lookup.update(state.polished_bullet_texts)

        assembled_sections: list[ResumeSection] = []
        for section in state.source_sections:
            assembled_entries: list[ResumeEntry] = []
            for entry in section.entries:
                active_bullet_ids = [bullet_id for bullet_id in entry.bullet_ids if bullet_id in selected_set]
                if not active_bullet_ids:
                    continue

                bullet_texts = []
                for bullet_id, source_text in zip(entry.bullet_ids, entry.bullets):
                    if bullet_id not in selected_set:
                        continue
                    bullet_texts.append(text_lookup.get(bullet_id, source_text))

                relevance = max(
                    (score_lookup.get(bullet_id, 0.0) for bullet_id in active_bullet_ids),
                    default=0.0,
                )
                assembled_entries.append(
                    entry.model_copy(
                        update={
                            "bullets": bullet_texts,
                            "bullet_ids": active_bullet_ids,
                            "relevance_score": relevance,
                        }
                    )
                )

            if assembled_entries:
                assembled_sections.append(
                    section.model_copy(update={"entries": assembled_entries})
                )

        return assembled_sections

    def _capture_polished_bullet_texts(self, state: PipelineState) -> PipelineState:
        """Align quality-polished pruner text back onto the deterministic bullet IDs."""
        if not state.pruned_sections or not state.selected_bullet_ids:
            return state

        selected_set = set(state.selected_bullet_ids)
        source_entry_map = self._source_entry_map(state)
        polished_texts = dict(state.polished_bullet_texts)
        mismatch_found = False

        for section in state.pruned_sections:
            for entry in section.entries:
                entry_id = entry.entry_id or self._match_source_entry_id(section.heading, entry, source_entry_map)
                if not entry_id or entry_id not in source_entry_map:
                    mismatch_found = True
                    continue

                _, source_entry = source_entry_map[entry_id]
                expected_ids = [bullet_id for bullet_id in source_entry.bullet_ids if bullet_id in selected_set]
                if len(expected_ids) != len(entry.bullets):
                    mismatch_found = True
                    continue

                for bullet_id, bullet_text in zip(expected_ids, entry.bullets):
                    polished_texts[bullet_id] = bullet_text

        if mismatch_found:
            logger.warning("  ⚠ Could not fully align pruner text to source bullet IDs; preserving existing draft text")

        state = state.model_copy(update={"polished_bullet_texts": polished_texts})
        rebuilt_pruned = self._assemble_sections_from_selection(state, state.selected_bullet_ids)
        return state.model_copy(update={"pruned_sections": rebuilt_pruned})

    def _source_entry_map(self, state: PipelineState) -> dict[str, tuple[str, ResumeEntry]]:
        return {
            entry.entry_id: (self._section_kind(section.heading), entry)
            for section in state.source_sections
            for entry in section.entries
            if entry.entry_id
        }

    def _match_source_entry_id(
        self,
        section_heading: str,
        entry: ResumeEntry,
        source_entry_map: dict[str, tuple[str, ResumeEntry]],
    ) -> str | None:
        """Best-effort lookup when an LLM response omits entry_id."""
        target_kind = self._section_kind(section_heading)
        title_key = self._normalize_text_key(entry.title)
        org_key = self._normalize_text_key(entry.organization)

        for source_kind, source_entry in source_entry_map.values():
            if source_kind != target_kind:
                continue
            if self._normalize_text_key(source_entry.title) == title_key and self._normalize_text_key(source_entry.organization) == org_key:
                return source_entry.entry_id
            if self._normalize_text_key(source_entry.title) == title_key and not org_key:
                return source_entry.entry_id
        return None

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
                    ),
                    "polished_bullet_texts": {},
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
            and state.source_bullets
        ):
            logger.info(
                "  ⚠ Underfill detected: %.0f%% fill — expanding from source inventory",
                underfill_metrics.fill_ratio * 100,
            )
            print(
                f"\n  ⚠  Page underfilled ({underfill_metrics.fill_ratio * 100:.0f}% fill) "
                f"— restoring content from source inventory"
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

    def _enforce_project_retention(self, state: PipelineState) -> PipelineState:
        """Restore missing source projects deterministically before rendering."""
        if not state.source_projects or not state.pruned_sections:
            return state

        missing_projects = self._find_missing_projects(
            state.source_projects,
            self._get_project_entries(state.pruned_sections),
        )
        if not missing_projects:
            return state

        missing_labels = ", ".join(
            self._format_entry_label(entry) for entry in missing_projects
        )
        logger.warning(
            "  ✖ Project retention guardrail restoring missing projects deterministically: %s",
            missing_labels,
        )
        print(
            "\n  ✖  Project retention guardrail restoring missing project(s): "
            f"{missing_labels or 'source projects'}"
        )
        selected = list(state.selected_bullet_ids)
        score_lookup = {
            bullet.bullet_id: bullet.relevance_score for bullet in state.mapped_bullets
        }
        for project_entry in missing_projects:
            selected.extend(
                self._pick_top_bullets(
                    project_entry,
                    score_lookup,
                    target=min(max(_MIN_PROJECT_BULLETS, 1), len(project_entry.bullet_ids)),
                )
            )

        ordered_selected = self._sort_selected_bullet_ids(state, selected)
        next_state = state.model_copy(update={"selected_bullet_ids": ordered_selected})
        rebuilt_sections = self._assemble_sections_from_selection(next_state, ordered_selected)
        state = next_state.model_copy(update={"pruned_sections": rebuilt_sections})

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

            candidates = self._enumerate_prune_candidates(current_state)
            if not candidates:
                logger.warning("  ⚠ Deterministic prune stopped: no removable units remain")
                break

            options = []

            for candidate in candidates:
                candidate_state = self._apply_prune_candidate(current_state, candidate)
                candidate_metrics = self.formatter.measure_render(candidate_state)
                if not self._candidate_improves(current_metrics, candidate_metrics):
                    continue
                options.append((candidate, candidate_state, candidate_metrics))

            if not options:
                logger.warning("  ⚠ Deterministic prune stopped: no candidate improved overflow")
                break

            best_candidate, best_state, best_metrics = self._select_best_candidate(
                options
            )

            current_state = best_state.model_copy(
                update={
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
        """Restore omitted source bullets one unit at a time until the page fills."""
        current_state = state
        current_metrics = metrics

        if not state.source_bullets:
            return state

        for step in range(1, _MAX_EXPAND_STEPS + 1):
            if current_metrics.fill_ratio >= _UNDERFILL_THRESHOLD:
                return current_state

            candidates = self._enumerate_expand_candidates(current_state)
            if not candidates:
                logger.info("  ✔ No more content to restore from source inventory")
                break

            for candidate in candidates:
                expanded_state = self._apply_expand_candidate(current_state, candidate)
                expanded_metrics = self.formatter.measure_render(expanded_state)

                if expanded_metrics.page_count > 1:
                    continue

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
                logger.info("  ✔ Expansion stopped: all remaining candidates would overflow")
                break

        return current_state

    def _enumerate_expand_candidates(self, state: PipelineState) -> list[dict]:
        """Return omitted source bullets sorted by mapped relevance."""
        candidates: list[dict] = []

        selected_set = set(state.selected_bullet_ids)
        score_lookup = {
            bullet.bullet_id: bullet.relevance_score for bullet in state.mapped_bullets
        }

        for source_bullet in state.source_bullets:
            if source_bullet.bullet_id in selected_set:
                continue
            candidates.append(
                {
                    "bullet_id": source_bullet.bullet_id,
                    "entry_id": source_bullet.entry_id,
                    "relevance": score_lookup.get(source_bullet.bullet_id, 0.0),
                    "label": f"restore bullet in {source_bullet.title or source_bullet.section_heading}",
                }
            )

        candidates.sort(key=lambda candidate: (-candidate["relevance"], candidate["bullet_id"]))
        return candidates

    def _apply_expand_candidate(self, state: PipelineState, candidate: dict) -> PipelineState:
        """Apply one expansion by adding a bullet ID back into the deterministic selection."""
        selected = list(state.selected_bullet_ids)
        if candidate["bullet_id"] not in selected:
            selected.append(candidate["bullet_id"])
        ordered_selected = self._sort_selected_bullet_ids(state, selected)
        next_state = state.model_copy(update={"selected_bullet_ids": ordered_selected})
        rebuilt_sections = self._assemble_sections_from_selection(next_state, ordered_selected)
        return next_state.model_copy(update={"pruned_sections": rebuilt_sections})

    def _enumerate_prune_candidates(self, state: PipelineState) -> list[_PruneCandidate]:
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
        if not state.source_sections:
            return candidates

        score_lookup = {
            bullet.bullet_id: bullet.relevance_score for bullet in state.mapped_bullets
        }
        selected_set = set(state.selected_bullet_ids)

        for section in state.source_sections:
            section_kind = self._section_kind(section.heading)
            priority = _SECTION_PRIORITY.get(section_kind, 1.5)
            active_entries = [
                entry
                for entry in section.entries
                if any(bullet_id in selected_set for bullet_id in entry.bullet_ids)
            ]

            for entry in active_entries:
                active_bullet_ids = [
                    bullet_id for bullet_id in entry.bullet_ids if bullet_id in selected_set
                ]
                entry_relevance = self._entry_relevance(entry, score_lookup)

                if section_kind == "skills":
                    for bullet_id in reversed(active_bullet_ids):
                        candidates.append(
                            _PruneCandidate(
                                kind="skills_line",
                                entry_id=entry.entry_id or "",
                                bullet_id=bullet_id,
                                relevance=0.0,
                                section_priority=priority,
                                label=f"remove skills line from {section.heading}",
                            )
                        )
                    continue

                if section_kind == "education":
                    min_bullets = _MIN_EDUCATION_BULLETS
                elif section_kind == "projects":
                    min_bullets = _MIN_PROJECT_BULLETS
                elif section_kind == "work":
                    min_bullets = _MIN_WORK_BULLETS
                else:
                    min_bullets = 0

                for bullet_id in active_bullet_ids[min_bullets:]:
                    candidates.append(
                        _PruneCandidate(
                            kind=f"{section_kind}_bullet",
                            entry_id=entry.entry_id or "",
                            bullet_id=bullet_id,
                            relevance=score_lookup.get(bullet_id, entry_relevance),
                            section_priority=priority,
                            label=f"trim {section_kind} bullet from {entry.title}",
                        )
                    )

                if section_kind == "work" and len(active_entries) > 2:
                    candidates.append(
                        _PruneCandidate(
                            kind="work_entry",
                            entry_id=entry.entry_id or "",
                            bullet_id=None,
                            relevance=entry_relevance,
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

    def _apply_prune_candidate(self, state: PipelineState, candidate: _PruneCandidate) -> PipelineState:
        """Apply one local trim to the selected bullet ID set and rebuild the draft."""
        source_entry_map = self._source_entry_map(state)
        selected = list(state.selected_bullet_ids)

        if candidate.kind == "work_entry":
            _, source_entry = source_entry_map[candidate.entry_id]
            selected = [
                bullet_id for bullet_id in selected if bullet_id not in set(source_entry.bullet_ids)
            ]
        elif candidate.bullet_id is not None:
            selected = [bullet_id for bullet_id in selected if bullet_id != candidate.bullet_id]

        ordered_selected = self._sort_selected_bullet_ids(state, selected)
        next_state = state.model_copy(update={"selected_bullet_ids": ordered_selected})
        rebuilt_sections = self._assemble_sections_from_selection(next_state, ordered_selected)
        return next_state.model_copy(update={"pruned_sections": rebuilt_sections})

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
        return section_kind(heading)

    @staticmethod
    def _sort_selected_bullet_ids(state: PipelineState, bullet_ids: list[str]) -> list[str]:
        """Return selected bullet IDs in canonical source order."""
        source_order = {bullet.bullet_id: bullet.order for bullet in state.source_bullets}
        deduped = list(dict.fromkeys(bullet_ids))
        return sorted(deduped, key=lambda bullet_id: source_order.get(bullet_id, 10**9))

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", " ", (text or "")).strip().lower()

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
