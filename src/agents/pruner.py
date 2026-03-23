"""Pruner Agent — removes redundancy and enforces conciseness.

Supports two modes:
  1. Standard mode: initial pruning pass from mapped_sections
  2. Overflow mode: aggressive condensing when the rendered PDF exceeds 1 page
     (triggered when state.overflow_pages is set)
"""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import PipelineState, ResumeSection


class PrunerAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "pruner"

    def _build_prompt(self, state: PipelineState) -> str:
        # In overflow mode, prune from pruned_sections (re-prune)
        # In standard mode, prune from mapped_sections (initial prune)
        if state.overflow_pages is not None and state.pruned_sections:
            source = state.pruned_sections
        elif state.mapped_sections:
            source = state.mapped_sections
        else:
            source = []

        sections_json = (
            "[" + ", ".join(s.model_dump_json() for s in source) + "]"
            if source
            else "[]"
        )

        analysis_json = (
            state.analysis.model_dump_json(indent=2) if state.analysis else "{}"
        )

        parts = [
            "Below is a tailored resume draft (as JSON sections) and the JD analysis.\n"
            "Prune for conciseness, remove low-relevance entries, "
            "enforce action verbs, and cut filler.\n\n"
            "=== JD ANALYSIS ===\n"
            f"{analysis_json}\n"
            "=== END JD ANALYSIS ===\n\n"
            "=== RESUME DRAFT ===\n"
            f"{sections_json}\n"
            "=== END RESUME DRAFT ==="
        ]

        # Inject overflow feedback for aggressive cutting
        if state.overflow_pages is not None:
            overflow_pct = int((state.overflow_pages - 1.0) * 100)
            parts.append(
                f"\n\n=== CRITICAL: PAGE OVERFLOW DETECTED ==="
                f"\nThe current draft renders to {state.overflow_pages:.1f} pages. "
                f"It is approximately {overflow_pct}% too long."
                f"\nYou MUST aggressively cut content to fit on exactly 1 page:"
                f"\n- Remove the LEAST relevant entry or role entirely"
                f"\n- Cut bullets down to 2-3 per role maximum"
                f"\n- Shorten every bullet to under 110 characters"
                f"\n- Merge or eliminate any redundant points"
                f"\n- Trim the Skills section to essential items only"
                f"\n- If the education section has more than 3 bullets, cut to 2"
                f"\nThis is iteration {state.render_iteration} of the overflow loop. "
                f"Be MORE aggressive than the previous pass."
                f"\n=== END OVERFLOW INSTRUCTIONS ==="
            )

        return "\n".join(parts)

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        sections = [ResumeSection.model_validate(s) for s in data]
        return state.model_copy(update={"pruned_sections": sections})
