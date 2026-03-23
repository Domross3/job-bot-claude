"""Pruner Agent — removes redundancy and enforces conciseness."""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import PipelineState, ResumeSection


class PrunerAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "pruner"

    def _build_prompt(self, state: PipelineState) -> str:
        # Serialize mapped sections to JSON for the LLM
        sections_json = (
            "[" + ", ".join(s.model_dump_json() for s in state.mapped_sections) + "]"
            if state.mapped_sections
            else "[]"
        )

        analysis_json = (
            state.analysis.model_dump_json(indent=2) if state.analysis else "{}"
        )

        return (
            "Below is a tailored resume draft (as JSON sections) and the JD analysis.\n"
            "Prune for conciseness, remove low-relevance entries, "
            "enforce action verbs, and cut filler.\n\n"
            "=== JD ANALYSIS ===\n"
            f"{analysis_json}\n"
            "=== END JD ANALYSIS ===\n\n"
            "=== RESUME DRAFT ===\n"
            f"{sections_json}\n"
            "=== END RESUME DRAFT ==="
        )

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        sections = [ResumeSection.model_validate(s) for s in data]
        return state.model_copy(update={"pruned_sections": sections})
