"""Semantic Mapper Agent — selects and rewords resume bullets to mirror the JD."""

from __future__ import annotations

import json

from ..agent import BaseAgent
from ..state import PipelineState, ResumeSection


class MapperAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "mapper"

    def _build_prompt(self, state: PipelineState) -> str:
        parts = [
            "Below is a structured analysis of a target Job Description, "
            "followed by a candidate's master resume.\n",
            "Your task: select the most relevant bullets from the master resume "
            "and tailor their wording to mirror the JD's vocabulary. "
            "Do NOT fabricate any experience.\n",
            "=== JD ANALYSIS ===",
            state.analysis.model_dump_json(indent=2) if state.analysis else "{}",
            "=== END JD ANALYSIS ===\n",
            "=== MASTER RESUME ===",
            state.master_resume,
            "=== END MASTER RESUME ===",
        ]

        # If this is a revision loop, inject Critic feedback as constraints
        if state.evaluation and not state.evaluation.approved:
            parts.append("\n=== CRITIC FEEDBACK (you MUST address these issues) ===")
            parts.append(f"Factual drift issues: {json.dumps(state.evaluation.factual_drift_issues)}")
            parts.append(f"Missing keywords: {json.dumps(state.evaluation.missing_keywords)}")
            parts.append(f"Suggestions: {json.dumps(state.evaluation.suggestions)}")
            parts.append("=== END CRITIC FEEDBACK ===")

            # Include the current draft so the mapper can expand on it
            # rather than starting from scratch and potentially producing less
            current_draft = state.pruned_sections or state.mapped_sections
            if current_draft:
                draft_json = "[" + ", ".join(s.model_dump_json() for s in current_draft) + "]"
                parts.append("\n=== CURRENT DRAFT (expand on this, do NOT produce less content) ===")
                parts.append(draft_json)
                parts.append("=== END CURRENT DRAFT ===")

        return "\n".join(parts)

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        sections = [ResumeSection.model_validate(s) for s in data]
        return state.model_copy(update={"mapped_sections": sections})
