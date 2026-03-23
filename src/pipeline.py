"""Pipeline orchestrator — runs agents in sequence with HITL gate."""

from __future__ import annotations

import logging
import sys
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
    """Orchestrates the 5-agent resume tailoring pipeline."""

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
        """Execute the full pipeline with human-in-the-loop gate."""

        state = PipelineState(
            master_resume=master_resume,
            job_description=job_description,
        )

        # ── Step 1: Analyze the JD ───────────────────────────────
        state = self.analyzer.run(state)
        self._print_analysis_summary(state)

        # ── Steps 2–4: Mapper → Pruner → Critic (with revision loop) ──
        state = self._run_core_loop(state)

        # ── HITL Gate: pause for human review ────────────────────
        action = self._human_review_gate(state)

        while action == "revise":
            if state.revision_count >= state.max_revisions:
                print(
                    f"\n⚠  Max revisions ({state.max_revisions}) reached. "
                    "Proceeding with best draft."
                )
                break
            state = state.model_copy(
                update={"revision_count": state.revision_count + 1}
            )
            state = self._run_core_loop(state)
            action = self._human_review_gate(state)

        if action == "reject":
            print("\n✖  Pipeline aborted by user.")
            return state

        # ── Step 5: Format ───────────────────────────────────────
        state = self.formatter.run(state)

        # ── Write output ─────────────────────────────────────────
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(state.final_resume or "", encoding="utf-8")
            print(f"\n✔  Resume written to {output_path}")

        return state

    # ── Internal helpers ─────────────────────────────────────────

    def _run_core_loop(self, state: PipelineState) -> PipelineState:
        """Run Mapper → Pruner → Critic, with automatic retry if Critic rejects."""
        state = self.mapper.run(state)
        state = self.pruner.run(state)
        state = self.critic.run(state)

        # Auto-retry loop (up to max_revisions) if Critic rejects
        while (
            state.evaluation
            and not state.evaluation.approved
            and state.revision_count < state.max_revisions
        ):
            logger.info(
                "Critic rejected (score: %.2f) — auto-revising (%d/%d)",
                state.evaluation.overall_score,
                state.revision_count + 1,
                state.max_revisions,
            )
            state = state.model_copy(
                update={"revision_count": state.revision_count + 1}
            )
            state = self.mapper.run(state)
            state = self.pruner.run(state)
            state = self.critic.run(state)

        return state

    def _human_review_gate(self, state: PipelineState) -> str:
        """Display the Critic evaluation + draft preview, then block for user input."""
        print("\n" + "=" * 70)
        print("  HUMAN REVIEW GATE — Pipeline paused for your approval")
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
        print("  DRAFT PREVIEW")
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
        """Print a brief summary of the JD analysis for user awareness."""
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
