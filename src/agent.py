"""Base agent class — thin wrapper around the Anthropic API."""

from __future__ import annotations

import json
import logging
import os
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
ENABLE_VISIBLE_REASONING = os.getenv("ENABLE_VISIBLE_REASONING", "").lower() in {"1", "true", "yes", "on"}
ENABLE_PROMPT_CACHING = os.getenv("ENABLE_PROMPT_CACHING", "1").lower() in {"1", "true", "yes", "on"}

_MODEL_PRICING: list[tuple[str, dict[str, float]]] = [
    (
        "claude-opus-4-6",
        {"input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
    ),
    (
        "claude-opus-4-5",
        {"input": 5.0, "cache_write": 6.25, "cache_read": 0.50, "output": 25.0},
    ),
    (
        "claude-sonnet-4-6",
        {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    ),
    (
        "claude-sonnet-4-5",
        {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    ),
    (
        "claude-sonnet-4",
        {"input": 3.0, "cache_write": 3.75, "cache_read": 0.30, "output": 15.0},
    ),
    (
        "claude-haiku-4-5",
        {"input": 1.0, "cache_write": 1.25, "cache_read": 0.10, "output": 5.0},
    ),
]


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


@dataclass(frozen=True)
class UsageSummary:
    """Anthropic token/cost accounting for one agent call."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    estimated_cost_usd: float | None
    duration_ms: int


def _effective_system_prompt(system_prompt: str) -> str:
    """Disable visible reasoning in production while preserving <answer> tags."""
    if ENABLE_VISIBLE_REASONING:
        return system_prompt

    return (
        f"{system_prompt.rstrip()}\n\n"
        "PRODUCTION MODE:\n"
        "- Do NOT include any <thinking> tags.\n"
        "- Do NOT include reasoning summaries.\n"
        "- Put your final output entirely inside <answer> tags.\n"
    )


def _pricing_for_model(model: str) -> dict[str, float] | None:
    for prefix, pricing in _MODEL_PRICING:
        if model.startswith(prefix):
            return pricing
    return None


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
        system_prompt=_effective_system_prompt(prompt_cfg["system_prompt"]),
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
        self.last_usage: UsageSummary | None = None

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
        user_content = self._build_message_content(state)
        raw_response = self._call_llm(user_content)
        tagged = self._extract_tagged_response(raw_response)
        if tagged.thinking and ENABLE_VISIBLE_REASONING:
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

    def _build_message_content(self, state: PipelineState) -> str | list[dict[str, Any]]:
        """Return either a single user string or structured content blocks."""
        return self._build_prompt(state)

    @staticmethod
    def _text_block(text: str, *, cache: bool = False) -> dict[str, Any]:
        """Build a Messages API text block, optionally marking it cacheable."""
        block: dict[str, Any] = {"type": "text", "text": text}
        if cache and ENABLE_PROMPT_CACHING and text.strip():
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def _call_llm(self, user_content: str | list[dict[str, Any]], retry_count: int = 0) -> str:
        """Call Anthropic API with exponential backoff (max 3 retries)."""
        max_retries = 3
        start = time.time()
        try:
            request_kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "system": self.config.system_prompt,
                "messages": [{"role": "user", "content": user_content}],
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
            input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
            cache_creation_input_tokens = int(
                getattr(response.usage, "cache_creation_input_tokens", 0) or 0
            )
            cache_read_input_tokens = int(
                getattr(response.usage, "cache_read_input_tokens", 0) or 0
            )
            duration_ms = int((time.time() - start) * 1000)
            estimated_cost_usd = self._estimate_cost_usd(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
            )
            self.last_usage = UsageSummary(
                model=self.config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_input_tokens=cache_creation_input_tokens,
                cache_read_input_tokens=cache_read_input_tokens,
                estimated_cost_usd=estimated_cost_usd,
                duration_ms=duration_ms,
            )
            logger.debug(
                "%s token usage: input=%d output=%d",
                self.config.name,
                input_tokens,
                output_tokens,
            )
            logger.info(
                "%s usage: input=%d cache_write=%d cache_read=%d output=%d est_cost=%s duration=%dms",
                self.config.name,
                input_tokens,
                cache_creation_input_tokens,
                cache_read_input_tokens,
                output_tokens,
                f"${estimated_cost_usd:.4f}" if estimated_cost_usd is not None else "n/a",
                duration_ms,
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
            return self._call_llm(user_content, retry_count + 1)

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
            return self._call_llm(user_content, retry_count + 1)

    def _estimate_cost_usd(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
    ) -> float | None:
        pricing = _pricing_for_model(self.config.model)
        if pricing is None:
            return None

        return (
            (input_tokens / 1_000_000) * pricing["input"]
            + (cache_creation_input_tokens / 1_000_000) * pricing["cache_write"]
            + (cache_read_input_tokens / 1_000_000) * pricing["cache_read"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

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
                "Return the corrected JSON entirely inside <answer> tags. "
                "Do not include markdown."
            )
            retry_raw = self._call_llm(fix_prompt)
            cleaned = self._extract_json(retry_raw)
            return json.loads(cleaned)  # Let it raise if still broken
