"""Pipeline orchestrator — runs agents in sequence with render-and-refine loop + HITL gate."""

from __future__ import annotations

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
        → Render-and-Refine loop (test render → overflow? → re-prune)
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
        """
        max_parity_retries = 2

        for attempt in range(max_parity_retries + 1):
            state = self.mapper.run(state)

            # ── DEBUG: Validate Mapper output has ≥3 bullets per Work Experience role ──
            if state.mapped_sections:
                for section in state.mapped_sections:
                    if "experience" in section.heading.lower() or "work" in section.heading.lower():
                        for entry in section.entries:
                            print(
                                f"DEBUG MAPPER OUTPUT: '{entry.title}' has "
                                f"{len(entry.bullets)} bullets"
                            )
                            if len(entry.bullets) < 3:
                                raise ValueError(
                                    f"Content floor violation: '{entry.title}' has only "
                                    f"{len(entry.bullets)} bullets (minimum 3 required). "
                                    f"Mapper prompt is not being followed."
                                )

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
        """Test render → if overflow → re-prune → repeat (max iterations).

        Returns state with confirmed single-page content.
        """
        logger.info("▶ Starting render-and-refine loop")

        for iteration in range(1, state.max_render_iterations + 1):
            state = self._enforce_project_retention(state)
            page_count = self.formatter.test_render(state)
            state = state.model_copy(update={"last_render_page_count": page_count})

            if page_count <= 1:
                logger.info("  ✔ Render fits in 1 page")
                # Clear any overflow metadata
                state = state.model_copy(
                    update={
                        "overflow_pages": None,
                        "render_iteration": 0,
                        "last_render_page_count": page_count,
                    }
                )
                return state

            # Overflow detected — estimate how far over
            overflow_estimate = page_count  # e.g. 2 pages = ~1.x actual
            # For a more precise estimate we'd measure content height,
            # but page_count is sufficient for the Pruner's instructions
            overflow_ratio = 1.0 + (0.3 * iteration)  # escalating estimate

            logger.warning(
                "  ⚠ Overflow: %d pages (iteration %d/%d) — re-pruning",
                page_count,
                iteration,
                state.max_render_iterations,
            )
            print(
                f"\n  ⚠  Render overflow: {page_count} pages — "
                f"re-pruning (iteration {iteration}/{state.max_render_iterations})"
            )

            state = state.model_copy(
                update={
                    "overflow_pages": overflow_ratio,
                    "render_iteration": iteration,
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

        # If we exhausted iterations, proceed with whatever we have
        state = self._enforce_project_retention(state)
        final_pages = self.formatter.test_render(state)
        if final_pages > 1:
            print(
                f"\n  ⚠  Could not fit to 1 page after {state.max_render_iterations} "
                f"iterations ({final_pages} pages). Proceeding with best draft."
            )
        state = state.model_copy(
            update={
                "overflow_pages": None,
                "render_iteration": 0,
                "last_render_page_count": final_pages,
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
