"""
tests/integration/test_etl_agent.py
──────────────────────────────────────
Integration tests for the ETL agent node (etl_node).

Tests the full agent function: state in → state update out.
Uses real temp files and real DuckDB/polars — no mocking.
Verifies recovery tiers, error capture, and ETLResult correctness.
"""

from __future__ import annotations

import json
import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.etl.agent import etl_node
from pipeline.core.schemas import (
    DataSource,
    DataSourceType,
    ETLResult,
    OutputFormat,
    PipelineStatus,
    RecoveryTier,
    SubTask,
    TaskComplexity,
    TaskPlan,
    TaskType,
)
from pipeline.core.state import PipelineState, initial_state


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sales_csv(tmp_path):
    p = tmp_path / "sales.csv"
    p.write_text(
        "date,revenue,units,region\n"
        + "\n".join(
            f"2024-01-{i:02d},{100 * i}.0,{i},{['north','south','east','west'][i % 4]}"
            for i in range(1, 21)
        )
    )
    return str(p)


@pytest.fixture
def metrics_parquet(tmp_path):
    p = tmp_path / "metrics.parquet"
    pl.DataFrame({
        "ts": [f"2024-01-{i:02d}" for i in range(1, 11)],
        "cpu": [float(i * 5) for i in range(1, 11)],
        "mem": [float(i * 0.8) for i in range(1, 11)],
    }).write_parquet(str(p))
    return str(p)


@pytest.fixture
def dirty_csv(tmp_path):
    p = tmp_path / "dirty.csv"
    p.write_text(
        "id,value,tag\n"
        "1,100,active\n"
        "2,NULL,inactive\n"
        "3,,\n"
        "4,200,active\n"
        "5,N/A,\n"
    )
    return str(p)


def _make_state(sources: list[DataSource], task_id: str = "test-task-001") -> PipelineState:
    """Build a PipelineState with a TaskPlan containing the given sources."""
    st = SubTask(subtask_id="st-etl", agent="etl", description="Load data")
    plan = TaskPlan(
        task_id=task_id,
        raw_task="Load and validate the data",
        data_sources=sources,
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[st],
        output_format=OutputFormat.MARKDOWN,
        complexity=TaskComplexity.LOW,
    )
    state: PipelineState = initial_state(task_id=task_id, raw_task="Load data")  # type: ignore[assignment]
    state["task_plan"] = plan
    state["status"] = PipelineStatus.RUNNING
    return state


# ─── Happy path ───────────────────────────────────────────────────────────────


class TestETLAgentHappyPath:
    def test_returns_dict(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert isinstance(update, dict)

    def test_etl_result_in_update(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert "etl_result" in update
        assert isinstance(update["etl_result"], ETLResult)

    def test_correct_row_count(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["etl_result"].row_count == 20

    def test_correct_column_count(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["etl_result"].column_count == 4

    def test_schema_populated(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        result: ETLResult = update["etl_result"]
        assert len(result.schema) == 4
        col_names = {s.name for s in result.schema}
        assert col_names == {"date", "revenue", "units", "region"}

    def test_data_json_populated(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["etl_result"].data_json is not None
        # Should be valid JSON
        parsed = json.loads(update["etl_result"].data_json)
        assert isinstance(parsed, (list, dict))

    def test_recovery_tier_none_on_clean_data(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["etl_result"].recovery_tier == RecoveryTier.NONE

    def test_elapsed_seconds_positive(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["etl_result"].elapsed_seconds > 0

    def test_status_running_after_success(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["status"] == PipelineStatus.RUNNING

    def test_parquet_source_loads(self, metrics_parquet):
        state = _make_state([DataSource(uri=metrics_parquet)])
        update = etl_node(state)
        assert update["etl_result"].row_count == 10
        assert update["etl_result"].column_count == 3


class TestETLAgentDirtyData:
    def test_dirty_data_still_loads(self, dirty_csv):
        state = _make_state([DataSource(uri=dirty_csv)])
        update = etl_node(state)
        result = update["etl_result"]
        assert result.row_count > 0

    def test_validation_issues_captured(self, dirty_csv):
        state = _make_state([DataSource(uri=dirty_csv)])
        update = etl_node(state)
        result = update["etl_result"]
        # Null values should produce at least one warning
        assert len(result.validation_issues) >= 1

    def test_no_errors_list_on_clean_run(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update.get("errors", []) == []


# ─── No-source edge case ──────────────────────────────────────────────────────


class TestETLAgentNoSources:
    def test_no_sources_returns_empty_result(self):
        state = _make_state([])
        update = etl_node(state)
        assert update["etl_result"].row_count == 0
        assert update["etl_result"].column_count == 0

    def test_no_sources_status_running(self):
        state = _make_state([])
        update = etl_node(state)
        assert update["status"] == PipelineStatus.RUNNING


# ─── Missing task plan ────────────────────────────────────────────────────────


class TestETLAgentNoTaskPlan:
    def test_no_task_plan_produces_degraded_result(self):
        state: PipelineState = initial_state(task_id="t1", raw_task="test")  # type: ignore[assignment]
        # task_plan is None by default
        update = etl_node(state)
        assert update["etl_result"].recovery_tier == RecoveryTier.DEGRADED

    def test_no_task_plan_error_recorded(self):
        state: PipelineState = initial_state(task_id="t1", raw_task="test")  # type: ignore[assignment]
        update = etl_node(state)
        assert len(update.get("errors", [])) >= 1

    def test_no_task_plan_status_partial(self):
        state: PipelineState = initial_state(task_id="t1", raw_task="test")  # type: ignore[assignment]
        update = etl_node(state)
        assert update["status"] == PipelineStatus.PARTIAL


# ─── Failure recovery ────────────────────────────────────────────────────────


class TestETLAgentRecovery:
    def test_missing_file_produces_degraded_result(self):
        source = DataSource(uri="/nonexistent/path/data.csv", source_type=DataSourceType.CSV)
        state = _make_state([source])
        update = etl_node(state)
        result = update["etl_result"]
        # Agent should not crash — recovery tier indicates degradation
        assert result.recovery_tier in (RecoveryTier.SIMPLIFIED, RecoveryTier.DEGRADED)

    def test_missing_file_error_recorded_in_update(self):
        source = DataSource(uri="/nonexistent/path/data.csv", source_type=DataSourceType.CSV)
        state = _make_state([source])
        update = etl_node(state)
        assert len(update.get("errors", [])) >= 1

    def test_missing_file_does_not_raise(self):
        source = DataSource(uri="/nonexistent/data.csv")
        state = _make_state([source])
        # Should return a dict, never raise
        try:
            update = etl_node(state)
            assert isinstance(update, dict)
        except Exception as e:
            pytest.fail(f"etl_node raised unexpectedly: {e}")

    def test_current_agent_cleared_after_run(self, sales_csv):
        state = _make_state([DataSource(uri=sales_csv)])
        update = etl_node(state)
        assert update["current_agent"] is None
