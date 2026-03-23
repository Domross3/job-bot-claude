"""Formatter Agent — renders the approved draft as clean Markdown."""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import PipelineState


class FormatterAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "formatter"

    def _build_prompt(self, state: PipelineState) -> str:
        pruned_json = (
            "[" + ", ".join(s.model_dump_json() for s in state.pruned_sections) + "]"
            if state.pruned_sections
            else "[]"
        )

        return (
            "Render the following structured resume data as a clean, "
            "professional Markdown resume.\n\n"
            "=== RESUME DATA (JSON) ===\n"
            f"{pruned_json}\n"
            "=== END RESUME DATA ==="
        )

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        # Formatter returns raw Markdown — no JSON parsing needed
        return state.model_copy(update={"final_resume": raw.strip()})
