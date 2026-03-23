"""Base agent class — thin wrapper around the Anthropic API."""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import yaml

from .state import PipelineState

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
MODELS_PATH = CONFIG_DIR / "models.yaml"
PROMPTS_DIR = CONFIG_DIR / "prompts"


@dataclass(frozen=True)
class AgentConfig:
    """Resolved configuration for a single agent."""

    name: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str


@dataclass(frozen=True)
class TaggedResponse:
    """Structured view of a tagged LLM response."""

    thinking: str
    answer: str


def load_agent_config(agent_name: str) -> AgentConfig:
    """Build an AgentConfig by merging models.yaml + prompts/<name>.yaml."""
    with open(MODELS_PATH) as f:
        all_models = yaml.safe_load(f)

    model_cfg = all_models.get(agent_name)
    if model_cfg is None:
        raise ValueError(
            f"No model config for agent '{agent_name}' in {MODELS_PATH}"
        )

    prompt_path = PROMPTS_DIR / f"{agent_name}.yaml"
    with open(prompt_path) as f:
        prompt_cfg = yaml.safe_load(f)

    return AgentConfig(
        name=agent_name,
        model=model_cfg["model"],
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_tokens", 4096),
        system_prompt=prompt_cfg["system_prompt"],
    )


class BaseAgent(ABC):
    """
    Every agent subclass must implement:
      - _build_prompt(state)  → str   (the user message for the LLM)
      - _parse_and_update(state, raw_response) → PipelineState
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        if config is None:
            config = load_agent_config(self.agent_name)
        self.config = config
        self.client = anthropic.Anthropic()

    # ── Subclass contract ────────────────────────────────────────

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Must match the key in models.yaml / prompts/<name>.yaml."""

    @abstractmethod
    def _build_prompt(self, state: PipelineState) -> str:
        """Assemble the user-role message from the current pipeline state."""

    @abstractmethod
    def _parse_and_update(
        self, state: PipelineState, raw: str
    ) -> PipelineState:
        """Parse the LLM's text response and return an updated state."""

    # ── Public API ───────────────────────────────────────────────

    def run(self, state: PipelineState) -> PipelineState:
        """Execute the agent: build prompt → call LLM → parse → update state."""
        logger.info("▶ Running %s agent [%s]", self.config.name, self.config.model)
        user_message = self._build_prompt(state)
        raw_response = self._call_llm(user_message)
        tagged = self._extract_tagged_response(raw_response)
        if tagged.thinking:
            logger.info("%s reasoning summary:\n%s", self.config.name, tagged.thinking)
            print(
                f"\n  [{self.config.name} reasoning summary]\n"
                f"  {tagged.thinking.replace(chr(10), chr(10) + '  ')}"
            )
        parseable_response = tagged.answer or raw_response
        updated_state = self._parse_and_update(state, parseable_response)
        logger.info("✔ %s agent complete", self.config.name)
        return updated_state

    # ── LLM interaction ──────────────────────────────────────────

    def _call_llm(self, user_message: str, retry_count: int = 0) -> str:
        """Call Anthropic API with exponential backoff (max 3 retries)."""
        max_retries = 3
        try:
            request_kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.config.system_prompt,
                "messages": [{"role": "user", "content": user_message}],
            }
            # NOTE: Adaptive thinking (API-level) is NOT used. We rely on
            # prompt-level <thinking>/<answer> tags instead, which work on
            # all models including Haiku. The tag extraction in run() handles
            # parsing the response.

            response = self.client.messages.create(
                **request_kwargs,
            )
            text_parts: list[str] = []
            for block in response.content:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_parts.append(block.text)
                elif block_type == "thinking":
                    logger.debug(
                        "%s returned an internal thinking block (%d chars)",
                        self.config.name,
                        len(getattr(block, "thinking", "")),
                    )
                elif block_type == "redacted_thinking":
                    logger.debug(
                        "%s returned a redacted thinking block",
                        self.config.name,
                    )

            text = "\n".join(part for part in text_parts if part).strip()
            logger.debug(
                "%s token usage: input=%d output=%d",
                self.config.name,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return text

        except anthropic.RateLimitError:
            if retry_count >= max_retries:
                raise
            wait = 2 ** (retry_count + 1)
            logger.warning(
                "Rate limited on %s — retrying in %ds (attempt %d/%d)",
                self.config.name,
                wait,
                retry_count + 1,
                max_retries,
            )
            time.sleep(wait)
            return self._call_llm(user_message, retry_count + 1)

        except anthropic.APIError as e:
            if retry_count >= max_retries:
                raise
            wait = 2 ** (retry_count + 1)
            logger.warning(
                "API error on %s: %s — retrying in %ds (attempt %d/%d)",
                self.config.name,
                e,
                wait,
                retry_count + 1,
                max_retries,
            )
            time.sleep(wait)
            return self._call_llm(user_message, retry_count + 1)

    # ── JSON helpers ─────────────────────────────────────────────

    @staticmethod
    def _extract_json(raw: str) -> str:
        """Strip markdown code fences if the LLM wraps its JSON output."""
        text = BaseAgent._extract_tagged_response(raw).answer or raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.index("\n")
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[: -len("```")]
        return text.strip()

    @staticmethod
    def _extract_tagged_response(raw: str) -> TaggedResponse:
        """Return visible thinking summary and answer payload from a tagged response."""
        text = raw.strip()
        thinking_match = re.search(
            r"<thinking>\s*(.*?)\s*</thinking>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        answer_match = re.search(
            r"<answer>\s*(.*?)\s*</answer>",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        thinking = thinking_match.group(1).strip() if thinking_match else ""
        if answer_match:
            answer = answer_match.group(1).strip()
        elif thinking_match:
            answer = re.sub(
                r"<thinking>\s*.*?\s*</thinking>",
                "",
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ).strip()
        else:
            answer = text

        return TaggedResponse(thinking=thinking, answer=answer)

    def _parse_json(self, raw: str) -> Any:
        """Parse JSON from LLM output, with one retry on failure."""
        cleaned = self._extract_json(raw)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(
                "%s returned invalid JSON — requesting fix: %s",
                self.config.name,
                e,
            )
            fix_prompt = (
                "Your previous <answer> block was not valid JSON. "
                "Here is the error:\n\n"
                f"{e}\n\n"
                "Return an optional brief <thinking> summary plus the corrected "
                "JSON entirely inside <answer> tags."
            )
            retry_raw = self._call_llm(fix_prompt)
            cleaned = self._extract_json(retry_raw)
            return json.loads(cleaned)  # Let it raise if still broken
