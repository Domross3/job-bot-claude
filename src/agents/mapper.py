"""Semantic Mapper Agent — selects and rewords resume bullets to mirror the JD."""

from __future__ import annotations

import json

from ..agent import BaseAgent
from ..state import MappedBullet, PipelineState


class MapperAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "mapper"

    def _build_prompt(self, state: PipelineState) -> str:
        source_bullets_json = (
            "[" + ", ".join(b.model_dump_json() for b in state.source_bullets) + "]"
            if state.source_bullets
            else "[]"
        )

        parts = [
            "Below is a structured analysis of a target Job Description, "
            "followed by the full source bullet inventory parsed from the master resume.\n",
            "Your task: rewrite and score EVERY source bullet so deterministic code "
            "can decide what to keep later. Do NOT fabricate any experience.\n",
            "=== JD ANALYSIS ===",
            state.analysis.model_dump_json(indent=2) if state.analysis else "{}",
            "=== END JD ANALYSIS ===\n",
            "=== SOURCE BULLET INVENTORY ===",
            source_bullets_json,
            "=== END SOURCE BULLET INVENTORY ===",
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
                parts.append("\n=== CURRENT DRAFT (for context only; keep scoring all source bullets) ===")
                parts.append(draft_json)
                parts.append("=== END CURRENT DRAFT ===")

        return "\n".join(parts)

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        parsed = [MappedBullet.model_validate(item) for item in data]
        parsed_by_id = {item.bullet_id: item for item in parsed}

        completed: list[MappedBullet] = []
        missing_ids: list[str] = []
        for source_bullet in state.source_bullets:
            mapped = parsed_by_id.get(source_bullet.bullet_id)
            if mapped is None:
                missing_ids.append(source_bullet.bullet_id)
                mapped = MappedBullet(
                    bullet_id=source_bullet.bullet_id,
                    rewritten_text=source_bullet.text,
                    relevance_score=0.0,
                )
            completed.append(mapped)

        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            print(
                "\n  ⚠  Mapper omitted source bullet IDs; restoring deterministically: "
                f"{preview}{'...' if len(missing_ids) > 5 else ''}"
            )

        return state.model_copy(update={"mapped_bullets": completed})
