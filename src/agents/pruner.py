"""Pruner Agent — quality polish pass (wording, action verbs, filler removal).

Page fitting is handled entirely by the deterministic pruner in pipeline.py.
This agent must NOT remove bullets or entries for space reasons.  Its only job
is to improve the *quality* of content the Mapper produced.
"""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import PipelineState, ResumeSection


class PrunerAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "pruner"

    def _build_prompt(self, state: PipelineState) -> str:
        # Always polish from mapped_sections (the Mapper's full output).
        # Only fall back to pruned_sections for project-retention re-runs.
        if state.force_source_project_inventory and state.pruned_sections:
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
            "Polish the wording: enforce strong action verbs, cut filler, "
            "and compress phrasing — but do NOT remove bullets or entries.\n"
            "Preserve every entry_id and bullet_ids array exactly as provided.\n\n"
            "=== JD ANALYSIS ===\n"
            f"{analysis_json}\n"
            "=== END JD ANALYSIS ===\n\n"
            "=== RESUME DRAFT ===\n"
            f"{sections_json}\n"
            "=== END RESUME DRAFT ==="
        ]

        if state.pruner_feedback:
            feedback_lines = "\n".join(f"- {item}" for item in state.pruner_feedback)
            parts.append(
                "\n\n=== PROGRAMMATIC REJECTION FEEDBACK (you MUST fix every item) ===\n"
                f"{feedback_lines}\n"
                "=== END PROGRAMMATIC REJECTION FEEDBACK ==="
            )

        if state.force_source_project_inventory and state.source_projects:
            source_projects_json = (
                "[" + ", ".join(p.model_dump_json() for p in state.source_projects) + "]"
            )
            parts.append(
                "\n\n=== SOURCE PROJECT INVENTORY (ALL OF THESE MUST REMAIN) ===\n"
                f"{source_projects_json}\n"
                "=== END SOURCE PROJECT INVENTORY ==="
            )

        # No overflow injection — page fitting is deterministic, not LLM-driven.

        return "\n".join(parts)

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        sections = [ResumeSection.model_validate(s) for s in data]
        return state.model_copy(update={"pruned_sections": sections})
