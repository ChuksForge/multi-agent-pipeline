"""
orchestrator/graph.py
──────────────────────
LangGraph StateGraph definition for the multi-agent pipeline.

Graph structure:
  START → planner_node → supervisor_node
                              ↓ (conditional edge via route_next)
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
          etl_node     analysis_node    report_node
              └───────────────┼───────────────┘
                              ▼
                        supervisor_node
                              ▼ (route_next → "end")
                             END

Every agent routes back to supervisor after completion.
Supervisor decides retry / skip / advance / end.

Usage:
    from pipeline.orchestrator.graph import build_graph, run_pipeline

    graph = build_graph()
    result = await run_pipeline(graph, raw_task="Analyse sales.csv")
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from pipeline.agents.analysis.agent import analysis_node
from pipeline.agents.etl.agent import etl_node
from pipeline.agents.report.agent import report_node
from pipeline.core.exceptions import InvalidTaskError, TaskPlanningError
from pipeline.core.schemas import PipelineStatus
from pipeline.core.state import PipelineState, initial_state
from pipeline.middleware.logger import get_logger
from pipeline.orchestrator.supervisor import route_next, supervisor_node

logger = get_logger(__name__)


def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph ready to invoke.
    Call this once at application startup and reuse the graph object.
    """
    graph = StateGraph(PipelineState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    graph.add_node("planner",    _planner_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("etl",        etl_node)
    graph.add_node("analysis",   analysis_node)
    graph.add_node("report",     report_node)

    # ── Edges ──────────────────────────────────────────────────────────────
    # Entry: always start with planner
    graph.add_edge(START, "planner")

    # Planner always feeds supervisor
    graph.add_edge("planner", "supervisor")

    # All agents route back to supervisor after completion
    graph.add_edge("etl",      "supervisor")
    graph.add_edge("analysis", "supervisor")
    graph.add_edge("report",   "supervisor")

    # Supervisor uses route_next to decide next step
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "etl":      "etl",
            "analysis": "analysis",
            "report":   "report",
            "end":      END,
        },
    )

    return graph.compile()


def _planner_node(state: PipelineState) -> dict[str, Any]:
    """
    Wrapper node that calls the Task Planner and writes the TaskPlan to state.
    Catches all errors so the graph never crashes from planning failures.
    """
    from pipeline.planner.planner import plan_task

    task_id = state["task_id"]
    raw_task = state["raw_task"]

    try:
        plan = plan_task(raw_task=raw_task, task_id=task_id)
        logger.info(
            "planner_node_complete",
            task_id=task_id,
            task_type=plan.task_type.value,
            subtasks=len(plan.subtasks),
        )
        return {
            "task_plan": plan,
            "status": PipelineStatus.RUNNING,
        }

    except (InvalidTaskError, TaskPlanningError) as e:
        logger.error("planner_node_failed", task_id=task_id, error=str(e))
        return {
            "task_plan": None,
            "status": PipelineStatus.FAILED,
        }
    except Exception as e:
        logger.exception("planner_node_unexpected", task_id=task_id, error=str(e))
        return {
            "task_plan": None,
            "status": PipelineStatus.FAILED,
        }


# ─── High-level runner ────────────────────────────────────────────────────────


async def run_pipeline(
    graph: Any,
    raw_task: str,
    task_id: str | None = None,
) -> PipelineState:
    """
    Run the full pipeline for a given task string.

    Args:
        graph:    Compiled StateGraph from build_graph().
        raw_task: Natural language task description.
        task_id:  Override task_id (auto-generated if None).

    Returns:
        Final PipelineState after all agents have run.
    """
    final_task_id = task_id or str(uuid.uuid4())
    start_state = initial_state(task_id=final_task_id, raw_task=raw_task)

    logger.info("pipeline_start", task_id=final_task_id, raw_task=raw_task[:80])

    final_state: PipelineState = await graph.ainvoke(start_state)  # type: ignore[assignment]

    logger.info(
        "pipeline_end",
        task_id=final_task_id,
        status=final_state.get("status", PipelineStatus.FAILED).value,
        has_report=final_state.get("report_result") is not None,
    )

    return final_state


def run_pipeline_sync(
    graph: Any,
    raw_task: str,
    task_id: str | None = None,
) -> PipelineState:
    """
    Synchronous wrapper around run_pipeline.
    Use in scripts and tests that don't have an event loop.
    """
    import asyncio
    return asyncio.run(run_pipeline(graph, raw_task, task_id))
