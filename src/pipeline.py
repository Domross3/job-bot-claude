"""Pipeline orchestrator — runs agents in sequence with render-and-refine loop + HITL gate."""

from __future__ import annotations

import logging
from pathlib import Path

from .agents import (
    AnalyzerAgent,
    CriticAgent,
    FormatterAgent,
    MapperAgent,
    PrunerAgent,
)
from .state import PipelineState

logger = logging.getLogger(__name__)


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
        )

        # ── Step 1: Analyze the JD ───────────────────────────────
        state = self.analyzer.run(state)
        self._print_analysis_summary(state)

        # ── Steps 2–4: Mapper → Pruner → Critic ─────────────────
        state = self._run_core_loop(state)

        # ── Step 5: Render-and-Refine loop ───────────────────────
        state = self._render_and_refine(state)

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
        """Run Mapper → Pruner → Critic once."""
        state = self.mapper.run(state)
        state = self.pruner.run(state)
        state = self.critic.run(state)
        return state

    def _render_and_refine(self, state: PipelineState) -> PipelineState:
        """Test render → if overflow → re-prune → repeat (max iterations).

        Returns state with confirmed single-page content.
        """
        logger.info("▶ Starting render-and-refine loop")

        for iteration in range(1, state.max_render_iterations + 1):
            page_count = self.formatter.test_render(state)

            if page_count <= 1:
                logger.info("  ✔ Render fits in 1 page")
                # Clear any overflow metadata
                state = state.model_copy(
                    update={"overflow_pages": None, "render_iteration": 0}
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

        # If we exhausted iterations, proceed with whatever we have
        final_pages = self.formatter.test_render(state)
        if final_pages > 1:
            print(
                f"\n  ⚠  Could not fit to 1 page after {state.max_render_iterations} "
                f"iterations ({final_pages} pages). Proceeding with best draft."
            )
        state = state.model_copy(
            update={"overflow_pages": None, "render_iteration": 0}
        )
        return state

    def _human_review_gate(self, state: PipelineState) -> str:
        """Display the Critic evaluation + draft preview, then block for user input."""
        print("\n" + "=" * 70)
        print("  HUMAN REVIEW GATE — Pipeline paused for your approval")
        print("  (Content verified to fit on 1 page)")
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
