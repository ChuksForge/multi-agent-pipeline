"""
middleware/token_tracker.py
───────────────────────────
LangChain callback handler that intercepts every LLM call and records:
  - token usage (input + output)
  - USD cost (via cost_model.py)
  - latency (wall-clock ms)
  - which agent triggered the call

The tracker is wired in at the LLM constructor level, not at call sites,
so it is impossible for an agent to "forget" to log cost.

Usage:
    tracker = TokenTracker(task_id="abc123", agent_id="etl")
    llm = ChatAnthropic(model="...", callbacks=[tracker])
    # ... run your chain ...
    entries: list[CostEntry] = tracker.entries
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from pipeline.core.schemas import CostEntry
from pipeline.middleware.cost_model import compute_cost
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)


class TokenTracker(BaseCallbackHandler):
    """
    Thread-safe LangChain callback handler.

    One instance per agent per pipeline run.
    Collects CostEntry records that the orchestrator merges into PipelineState.cost_log.
    """

    def __init__(self, task_id: str, agent_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self.agent_id = agent_id
        self.entries: list[CostEntry] = []
        self._call_start_times: dict[str, float] = {}  # run_id → wall time

    # ── LangChain callback hooks ───────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Record wall-clock start time keyed by run_id."""
        self._call_start_times[str(run_id)] = time.monotonic()

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """
        Extract token usage from the LLMResult and write a CostEntry.

        LangChain surfaces token counts in two places depending on provider:
          1. response.llm_output["usage"] (Anthropic)
          2. response.generations[0][0].generation_info (some providers)
        We check both.
        """
        elapsed_ms = self._pop_elapsed(str(run_id))
        model_id = self._extract_model(response)
        input_tokens, output_tokens = self._extract_tokens(response)

        cost = compute_cost(model_id, input_tokens, output_tokens)

        entry = CostEntry(
            task_id=self.task_id,
            agent_id=self.agent_id,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=elapsed_ms,
            call_type="completion",
        )
        self.entries.append(entry)

        logger.debug(
            "token_usage_recorded",
            task_id=self.task_id,
            agent_id=self.agent_id,
            model=model_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=round(elapsed_ms, 1),
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        """Clean up timing entry on error — don't leave stale keys."""
        self._pop_elapsed(str(run_id))
        logger.warning(
            "llm_call_error",
            task_id=self.task_id,
            agent_id=self.agent_id,
            error=str(error),
        )

    # ── Aggregation helpers ───────────────────────────────────────────────────

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.entries)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.entries)

    @property
    def total_latency_ms(self) -> float:
        return sum(e.latency_ms for e in self.entries)

    def summary(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "calls": len(self.entries),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _pop_elapsed(self, run_id: str) -> float:
        start = self._call_start_times.pop(run_id, None)
        if start is None:
            return 0.0
        return (time.monotonic() - start) * 1000.0

    @staticmethod
    def _extract_model(response: LLMResult) -> str:
        """Pull model name from LLMResult metadata."""
        llm_output = response.llm_output or {}
        # Anthropic places it here
        model = llm_output.get("model") or llm_output.get("model_id", "")
        if model:
            return model
        # Fallback: check first generation's generation_info
        try:
            gen_info = response.generations[0][0].generation_info or {}
            return gen_info.get("model", "unknown")
        except (IndexError, AttributeError):
            return "unknown"

    @staticmethod
    def _extract_tokens(response: LLMResult) -> tuple[int, int]:
        """
        Return (input_tokens, output_tokens).
        Handles both Anthropic and OpenAI token field names.
        """
        llm_output = response.llm_output or {}

        # Anthropic usage block
        usage = llm_output.get("usage") or llm_output.get("token_usage") or {}
        input_tokens = (
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or 0
        )
        output_tokens = (
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or 0
        )

        # If still zero, try generation-level info (some LangChain versions)
        if input_tokens == 0 and output_tokens == 0:
            try:
                gen_info = response.generations[0][0].generation_info or {}
                usage2 = gen_info.get("usage", {})
                input_tokens = usage2.get("input_tokens", 0)
                output_tokens = usage2.get("output_tokens", 0)
            except (IndexError, AttributeError):
                pass

        return int(input_tokens), int(output_tokens)


# ─── Factory ──────────────────────────────────────────────────────────────────


def make_tracker(task_id: str, agent_id: str) -> TokenTracker:
    """Convenience factory — preferred over direct instantiation."""
    return TokenTracker(task_id=task_id, agent_id=agent_id)
