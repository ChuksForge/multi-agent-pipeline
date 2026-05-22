"""
core/state.py
─────────────
The single source of truth for a pipeline run.

All agents read from and write into this TypedDict.
LangGraph treats this as the graph state — each node receives the full
state and returns a partial update dict.

Design decisions:
  - All result fields are Optional so partial pipelines are valid
  - errors is append-only (List[AgentErrorRecord]) — never overwritten
  - cost_log is append-only — the TokenTracker middleware writes to it
  - status is the single authority on pipeline health
"""

from __future__ import annotations

from typing import Annotated, Any

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    CostEntry,
    ETLResult,
    PipelineStatus,
    ReportResult,
    TaskPlan,
)


def _append_list(existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer: always append, never overwrite. Used for errors and cost_log."""
    return existing + new


class PipelineState(TypedDict):
    """
    Shared state for the entire pipeline run.

    LangGraph node contract:
      - Each node receives the full PipelineState
      - Each node returns a dict with ONLY the keys it updates
      - LangGraph merges updates using field-level reducers

    Append-only fields use Annotated[list, _append_list] so concurrent
    nodes can safely add entries without stomping each other.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    task_id: str
    raw_task: str

    # ── Planning ──────────────────────────────────────────────────────────────
    task_plan: TaskPlan | None

    # ── Agent Results ─────────────────────────────────────────────────────────
    etl_result: ETLResult | None
    analysis_result: AnalysisResult | None
    report_result: ReportResult | None

    # ── Error Tracking (append-only) ──────────────────────────────────────────
    errors: Annotated[list[AgentErrorRecord], _append_list]

    # ── Cost Tracking (append-only) ───────────────────────────────────────────
    cost_log: Annotated[list[CostEntry], _append_list]

    # ── Orchestrator Control ──────────────────────────────────────────────────
    status: PipelineStatus
    current_agent: str | None       # Which agent is active right now
    retry_counts: dict[str, int]    # agent_id → attempt count
    skip_agents: list[str]          # Agents the orchestrator decided to skip

    # ── Routing Hints (set by orchestrator, read by conditional edges) ────────
    next_agent: str | None          # Explicit next-agent override


def initial_state(task_id: str, raw_task: str) -> PipelineState:
    """
    Factory for a fresh PipelineState.
    Always start from this — never construct the TypedDict manually.
    """
    return PipelineState(
        task_id=task_id,
        raw_task=raw_task,
        task_plan=None,
        etl_result=None,
        analysis_result=None,
        report_result=None,
        errors=[],
        cost_log=[],
        status=PipelineStatus.PENDING,
        current_agent=None,
        retry_counts={},
        skip_agents=[],
        next_agent=None,
    )


# ─── Convenience helpers ──────────────────────────────────────────────────────

def has_etl_data(state: PipelineState) -> bool:
    """True if ETL agent produced usable data (not just a degraded skeleton)."""
    return (
        state.get("etl_result") is not None
        and state["etl_result"].row_count > 0  # type: ignore[union-attr]
    )


def has_analysis_data(state: PipelineState) -> bool:
    return state.get("analysis_result") is not None


def is_failed(state: PipelineState) -> bool:
    return state["status"] == PipelineStatus.FAILED


def agent_retry_count(state: PipelineState, agent_id: str) -> int:
    return state.get("retry_counts", {}).get(agent_id, 0)


def increment_retry(state: PipelineState, agent_id: str) -> dict[str, Any]:
    """Return a partial state update that increments the retry counter."""
    counts = dict(state.get("retry_counts", {}))
    counts[agent_id] = counts.get(agent_id, 0) + 1
    return {"retry_counts": counts}


def mark_agent_skipped(state: PipelineState, agent_id: str) -> dict[str, Any]:
    """Return a partial state update that adds agent_id to skip list."""
    skipped = list(state.get("skip_agents", []))
    if agent_id not in skipped:
        skipped.append(agent_id)
    return {"skip_agents": skipped}
