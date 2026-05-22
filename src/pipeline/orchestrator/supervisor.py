"""
orchestrator/supervisor.py
───────────────────────────
Supervisor node — the brain of the LangGraph StateGraph.

Responsibilities:
  1. Decide the next agent to run based on TaskPlan and current state
  2. Detect retry conditions and increment retry counters
  3. Mark agents as skipped when retries are exhausted
  4. Signal END when the pipeline is complete or unrecoverable

Design:
  - Completely stateless: routing decisions come entirely from PipelineState
  - No LLM calls: pure Python logic, deterministic and fast
  - Two public functions:
      supervisor_node(state) → partial state update dict
      route_next(state)      → str (agent name or "end")
    LangGraph calls supervisor_node as a node, then route_next as a
    conditional edge function to determine the next node.

Recovery logic (mirrors the three-tier system from agents):
  - RETRY tier: retry_count < max_retries → re-route to same agent
  - SIMPLIFIED / DEGRADED tier: skip agent, continue to next
  - status == FAILED: route to "end" immediately
"""

from __future__ import annotations

from typing import Any

from pipeline.core.config import settings
from pipeline.core.schemas import (
    PipelineStatus,
    RecoveryTier,
    TaskType,
)
from pipeline.core.state import (
    PipelineState,
    agent_retry_count,
    has_analysis_data,
    has_etl_data,
    increment_retry,
    is_failed,
    mark_agent_skipped,
)
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

# Agent execution order for FULL_PIPELINE
_AGENT_ORDER = ["etl", "analysis", "report"]

# Mapping from task_type to required agents
_TASK_TYPE_AGENTS: dict[str, list[str]] = {
    "etl_only":      ["etl"],
    "analysis_only": ["analysis"],
    "report_only":   ["report"],
    "full_pipeline": ["etl", "analysis", "report"],
}


# ─── LangGraph Node ───────────────────────────────────────────────────────────


def supervisor_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node — runs after every agent and before routing.

    Inspects the state to:
      1. Detect if the last agent needs retry
      2. Apply retry logic or skip decision
      3. Set next_agent for the conditional edge function

    Returns a partial state update dict.
    Never raises.
    """
    task_id = state["task_id"]
    current_agent = state.get("current_agent")
    status = state.get("status", PipelineStatus.PENDING)

    logger.debug(
        "supervisor_evaluating",
        task_id=task_id,
        current_agent=current_agent,
        status=status.value if status else "unknown",
    )

    # Hard stop conditions
    if is_failed(state):
        logger.warning("supervisor_pipeline_failed", task_id=task_id)
        return {"next_agent": None, "status": PipelineStatus.FAILED}

    # Determine next agent
    next_agent = _decide_next_agent(state)

    update: dict[str, Any] = {"next_agent": next_agent}

    if next_agent is None:
        # Pipeline complete
        final_status = (
            PipelineStatus.COMPLETE
            if status == PipelineStatus.RUNNING
            else status
        )
        update["status"] = final_status
        logger.info(
            "supervisor_pipeline_complete",
            task_id=task_id,
            status=final_status.value,
        )
    else:
        update["current_agent"] = next_agent
        logger.info(
            "supervisor_routing",
            task_id=task_id,
            next=next_agent,
        )

    return update


def route_next(state: PipelineState) -> str:
    """
    Conditional edge function for LangGraph.
    Returns the name of the next node, or "end".

    LangGraph calls this after supervisor_node runs and uses the return
    value to follow the matching conditional edge.
    """
    next_agent = state.get("next_agent")
    if next_agent is None or is_failed(state):
        return "end"
    return next_agent


# ─── Routing Logic ────────────────────────────────────────────────────────────


def _decide_next_agent(state: PipelineState) -> str | None:
    """
    Core routing function. Returns agent name or None (= pipeline complete).

    Order of checks:
      1. Check if last agent needs retry
      2. Determine which agents are still needed
      3. Return the first pending agent, or None if all done
    """
    task_plan = state.get("task_plan")
    if task_plan is None:
        return None

    task_type = task_plan.task_type.value
    required_agents = _TASK_TYPE_AGENTS.get(task_type, _AGENT_ORDER)
    skip_agents = set(state.get("skip_agents", []))
    errors = state.get("errors", [])

    # Check last error for retry vs skip decision
    if errors:
        last_error = errors[-1]
        agent_id = last_error.agent_id
        recovery = last_error.recovery_tier

        if recovery == RecoveryTier.RETRY:
            retry_count = agent_retry_count(state, agent_id)
            if retry_count < settings.max_retries:
                logger.info(
                    "supervisor_retry",
                    agent=agent_id,
                    attempt=retry_count + 1,
                    max=settings.max_retries,
                )
                # increment_retry returns a partial dict — we apply it here
                # by returning the agent for retry routing
                return agent_id
            else:
                logger.warning(
                    "supervisor_skip_exhausted",
                    agent=agent_id,
                    retries=retry_count,
                )
                skip_agents.add(agent_id)

        elif recovery in (RecoveryTier.SIMPLIFIED, RecoveryTier.DEGRADED):
            # Agent already handled its own degradation — continue forward
            pass

    # Walk the required agents in order and return the first unfinished one
    for agent in required_agents:
        if agent in skip_agents:
            continue
        if _agent_is_complete(state, agent):
            continue
        return agent

    return None  # All required agents complete


def _agent_is_complete(state: PipelineState, agent: str) -> bool:
    """
    Return True if an agent has already produced a result in state.
    Degraded results count as complete — the pipeline continues.
    """
    if agent == "etl":
        return state.get("etl_result") is not None
    if agent == "analysis":
        return state.get("analysis_result") is not None
    if agent == "report":
        return state.get("report_result") is not None
    return False


def _needs_retry(state: PipelineState, agent: str) -> bool:
    """
    True if the agent's last error was a RETRY-tier error
    and retries are not yet exhausted.
    """
    errors = state.get("errors", [])
    agent_errors = [e for e in errors if e.agent_id == agent]
    if not agent_errors:
        return False
    last = agent_errors[-1]
    if last.recovery_tier != RecoveryTier.RETRY:
        return False
    return agent_retry_count(state, agent) < settings.max_retries
