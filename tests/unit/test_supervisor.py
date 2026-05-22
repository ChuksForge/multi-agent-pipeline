"""
tests/unit/test_supervisor.py
──────────────────────────────
Unit tests for the orchestrator supervisor.
Tests routing decisions, retry logic, skip logic, and end conditions.
No LLM calls — supervisor is deterministic pure Python.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    DataSource,
    ETLResult,
    OutputFormat,
    PipelineStatus,
    RecoveryTier,
    ReportResult,
    SubTask,
    TaskComplexity,
    TaskPlan,
    TaskType,
)
from pipeline.core.state import PipelineState, initial_state
from pipeline.orchestrator.supervisor import (
    _agent_is_complete,
    _decide_next_agent,
    route_next,
    supervisor_node,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_plan(task_type: TaskType = TaskType.FULL_PIPELINE,
               task_id: str = "sup-001") -> TaskPlan:
    subtasks = [
        SubTask(subtask_id="st-001", agent="etl", description="Load"),
        SubTask(subtask_id="st-002", agent="analysis", description="Analyse",
                depends_on=["st-001"]),
        SubTask(subtask_id="st-003", agent="report", description="Report",
                depends_on=["st-002"]),
    ]
    if task_type == TaskType.ETL_ONLY:
        subtasks = [subtasks[0]]
    elif task_type == TaskType.ANALYSIS_ONLY:
        subtasks = [subtasks[1]]

    return TaskPlan(
        task_id=task_id,
        raw_task="test task",
        data_sources=[DataSource(uri="data/test.csv")],
        task_type=task_type,
        subtasks=subtasks,
        output_format=OutputFormat.MARKDOWN,
        complexity=TaskComplexity.LOW,
    )


def _fresh_state(task_type: TaskType = TaskType.FULL_PIPELINE,
                 task_id: str = "sup-001") -> PipelineState:
    state: PipelineState = initial_state(task_id=task_id, raw_task="test")  # type: ignore[assignment]
    state["task_plan"] = _make_plan(task_type, task_id)
    state["status"] = PipelineStatus.RUNNING
    return state


def _with_etl_done(state: PipelineState) -> PipelineState:
    state["etl_result"] = ETLResult(
        task_id=state["task_id"], source_ids=[], row_count=100,
        column_count=3, schema=[],
    )
    return state


def _with_analysis_done(state: PipelineState) -> PipelineState:
    state["analysis_result"] = AnalysisResult(task_id=state["task_id"])
    return state


def _with_report_done(state: PipelineState) -> PipelineState:
    state["report_result"] = ReportResult(
        task_id=state["task_id"],
        output_format=OutputFormat.MARKDOWN,
        title="Test",
        full_content="# Report",
        word_count=2,
    )
    return state


def _with_error(state: PipelineState, agent: str,
                tier: RecoveryTier) -> PipelineState:
    record = AgentErrorRecord(
        agent_id=agent, attempt=1,
        error_type="TestError", message="test",
        recovery_tier=tier,
    )
    state["errors"] = list(state.get("errors", [])) + [record]
    return state


# ─── _agent_is_complete ───────────────────────────────────────────────────────


class TestAgentIsComplete:
    def test_etl_not_complete_initially(self):
        state = _fresh_state()
        assert _agent_is_complete(state, "etl") is False

    def test_etl_complete_after_result(self):
        state = _with_etl_done(_fresh_state())
        assert _agent_is_complete(state, "etl") is True

    def test_analysis_not_complete_initially(self):
        state = _fresh_state()
        assert _agent_is_complete(state, "analysis") is False

    def test_analysis_complete_after_result(self):
        state = _with_analysis_done(_fresh_state())
        assert _agent_is_complete(state, "analysis") is True

    def test_report_not_complete_initially(self):
        state = _fresh_state()
        assert _agent_is_complete(state, "report") is False

    def test_report_complete_after_result(self):
        state = _with_report_done(_fresh_state())
        assert _agent_is_complete(state, "report") is True

    def test_unknown_agent_not_complete(self):
        state = _fresh_state()
        assert _agent_is_complete(state, "unknown_agent") is False


# ─── _decide_next_agent ───────────────────────────────────────────────────────


class TestDecideNextAgent:
    def test_first_agent_is_etl_for_full_pipeline(self):
        state = _fresh_state()
        assert _decide_next_agent(state) == "etl"

    def test_analysis_after_etl_complete(self):
        state = _with_etl_done(_fresh_state())
        assert _decide_next_agent(state) == "analysis"

    def test_report_after_analysis_complete(self):
        state = _with_analysis_done(_with_etl_done(_fresh_state()))
        assert _decide_next_agent(state) == "report"

    def test_none_after_all_complete(self):
        state = _fresh_state()
        state = _with_etl_done(state)
        state = _with_analysis_done(state)
        state = _with_report_done(state)
        assert _decide_next_agent(state) is None

    def test_etl_only_pipeline_ends_after_etl(self):
        state = _fresh_state(TaskType.ETL_ONLY)
        state = _with_etl_done(state)
        assert _decide_next_agent(state) is None

    def test_no_task_plan_returns_none(self):
        state: PipelineState = initial_state(task_id="t1", raw_task="test")  # type: ignore[assignment]
        state["task_plan"] = None
        assert _decide_next_agent(state) is None

    def test_skipped_agent_bypassed(self):
        state = _fresh_state()
        state["skip_agents"] = ["etl"]
        # ETL is skipped → analysis is next
        result = _decide_next_agent(state)
        assert result == "analysis"

    def test_retry_tier_routes_back_to_same_agent(self):
        state = _fresh_state()
        # ETL has a RETRY-tier error and retries not exhausted
        state = _with_error(state, "etl", RecoveryTier.RETRY)
        state["retry_counts"] = {"etl": 0}
        result = _decide_next_agent(state)
        assert result == "etl"

    def test_retry_exhausted_skips_agent(self):
        state = _fresh_state()
        state = _with_error(state, "etl", RecoveryTier.RETRY)
        # max_retries is 3 — set count at limit
        state["retry_counts"] = {"etl": 3}
        result = _decide_next_agent(state)
        # ETL exhausted → moves to analysis
        assert result == "analysis"

    def test_degraded_tier_continues_forward(self):
        state = _fresh_state()
        # ETL ran but degraded — etl_result is set (degraded result counts)
        state = _with_etl_done(state)
        state = _with_error(state, "etl", RecoveryTier.DEGRADED)
        result = _decide_next_agent(state)
        assert result == "analysis"


# ─── supervisor_node ─────────────────────────────────────────────────────────


class TestSupervisorNode:
    def test_returns_dict(self):
        state = _fresh_state()
        update = supervisor_node(state)
        assert isinstance(update, dict)

    def test_next_agent_etl_initially(self):
        state = _fresh_state()
        update = supervisor_node(state)
        assert update.get("next_agent") == "etl"

    def test_next_agent_none_when_all_done(self):
        state = _fresh_state()
        state = _with_etl_done(state)
        state = _with_analysis_done(state)
        state = _with_report_done(state)
        update = supervisor_node(state)
        assert update.get("next_agent") is None

    def test_status_complete_when_pipeline_done(self):
        state = _fresh_state()
        state = _with_etl_done(state)
        state = _with_analysis_done(state)
        state = _with_report_done(state)
        update = supervisor_node(state)
        assert update.get("status") == PipelineStatus.COMPLETE

    def test_failed_state_returns_failed(self):
        state = _fresh_state()
        state["status"] = PipelineStatus.FAILED
        update = supervisor_node(state)
        assert update.get("status") == PipelineStatus.FAILED
        assert update.get("next_agent") is None

    def test_current_agent_set_to_next(self):
        state = _fresh_state()
        update = supervisor_node(state)
        assert update.get("current_agent") == "etl"


# ─── route_next ──────────────────────────────────────────────────────────────


class TestRouteNext:
    def test_returns_etl_when_next_agent_etl(self):
        state = _fresh_state()
        state["next_agent"] = "etl"
        assert route_next(state) == "etl"

    def test_returns_analysis(self):
        state = _fresh_state()
        state["next_agent"] = "analysis"
        assert route_next(state) == "analysis"

    def test_returns_report(self):
        state = _fresh_state()
        state["next_agent"] = "report"
        assert route_next(state) == "report"

    def test_returns_end_when_next_none(self):
        state = _fresh_state()
        state["next_agent"] = None
        assert route_next(state) == "end"

    def test_returns_end_when_failed(self):
        state = _fresh_state()
        state["status"] = PipelineStatus.FAILED
        state["next_agent"] = "etl"  # even if set, failed → end
        assert route_next(state) == "end"
