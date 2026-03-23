"""Analyzer Agent — extracts structured requirements from the Job Description."""

from __future__ import annotations

from ..agent import BaseAgent
from ..state import JDAnalysis, PipelineState


class AnalyzerAgent(BaseAgent):
    @property
    def agent_name(self) -> str:
        return "analyzer"

    def _build_prompt(self, state: PipelineState) -> str:
        return (
            "Analyze the following Job Description and extract structured data.\n\n"
            "=== JOB DESCRIPTION ===\n"
            f"{state.job_description}\n"
            "=== END ==="
        )

    def _parse_and_update(self, state: PipelineState, raw: str) -> PipelineState:
        data = self._parse_json(raw)
        analysis = JDAnalysis.model_validate(data)
        return state.model_copy(update={"analysis": analysis})
