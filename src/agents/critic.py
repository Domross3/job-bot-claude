"""Critic Agent — verifies factual accuracy and keyword coverage."""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import Evaluation, PipelineState


class CriticAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "critic"

    def _build_prompt(self, state: PipelineState) -> str:
        pruned_json = (
            "[" + ", ".join(s.model_dump_json() for s in state.pruned_sections) + "]"
            if state.pruned_sections
            else "[]"
        )

        analysis_json = (
            state.analysis.model_dump_json(indent=2) if state.analysis else "{}"
        )

        return (
            "Compare the tailored resume draft against the original master resume "
            "and the JD analysis. Check for factual drift and keyword coverage.\n\n"
            "=== JD ANALYSIS ===\n"
            f"{analysis_json}\n"
            "=== END JD ANALYSIS ===\n\n"
            "=== MASTER RESUME (original source of truth) ===\n"
            f"{state.master_resume}\n"
            "=== END MASTER RESUME ===\n\n"
            "=== TAILORED DRAFT (to evaluate) ===\n"
            f"{pruned_json}\n"
            "=== END TAILORED DRAFT ==="
        )

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        evaluation = Evaluation.model_validate(data)

        # Append to history for tracking across revision loops
        updated_history = state.critic_history + [evaluation]

        return state.model_copy(
            update={
                "evaluation": evaluation,
                "critic_history": updated_history,
            }
        )
