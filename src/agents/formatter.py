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

        # Extract the contact header from the master resume (everything before first ---)
        contact_block = ""
        if state.master_resume:
            lines = state.master_resume.split("\n")
            header_lines = []
            for line in lines:
                if line.strip().startswith("---"):
                    break
                header_lines.append(line)
            contact_block = "\n".join(header_lines).strip()

        return (
            "Render the following structured resume data as a clean, "
            "professional Markdown resume.\n\n"
            "=== CONTACT INFO (use this for the name and contact line) ===\n"
            f"{contact_block}\n"
            "=== END CONTACT INFO ===\n\n"
            "=== RESUME DATA (JSON) ===\n"
            f"{pruned_json}\n"
            "=== END RESUME DATA ==="
        )

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        # Formatter returns raw Markdown — no JSON parsing needed
        return state.model_copy(update={"final_resume": raw.strip()})
