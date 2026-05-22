"""
tests/unit/test_schemas.py
──────────────────────────
Tests for all Pydantic v2 domain models in core/schemas.py.
Validates field constraints, validators, computed properties, and factories.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    ColumnSchema,
    CostEntry,
    CostSummary,
    DataSource,
    DataSourceType,
    ETLResult,
    OutputFormat,
    RecoveryTier,
    SubTask,
    SummaryStats,
    TaskComplexity,
    TaskPlan,
    TaskType,
    ValidationIssue,
)


# ─── DataSource ───────────────────────────────────────────────────────────────


class TestDataSource:
    def test_auto_detects_csv(self):
        ds = DataSource(uri="data/sales.csv")
        assert ds.source_type == DataSourceType.CSV

    def test_auto_detects_parquet(self):
        ds = DataSource(uri="/tmp/metrics.parquet")
        assert ds.source_type == DataSourceType.PARQUET

    def test_auto_detects_json(self):
        ds = DataSource(uri="events.json")
        assert ds.source_type == DataSourceType.JSON

    def test_auto_detects_postgres(self):
        ds = DataSource(uri="postgresql://user:pass@host/db")
        assert ds.source_type == DataSourceType.POSTGRES

    def test_explicit_type_not_overridden(self):
        ds = DataSource(uri="data/file.csv", source_type=DataSourceType.PARQUET)
        assert ds.source_type == DataSourceType.PARQUET

    def test_empty_uri_raises(self):
        with pytest.raises(ValidationError, match="URI cannot be empty"):
            DataSource(uri="   ")

    def test_uri_is_stripped(self):
        ds = DataSource(uri="  data/sales.csv  ")
        assert ds.uri == "data/sales.csv"

    def test_source_id_auto_generated(self):
        ds = DataSource(uri="data/a.csv")
        assert len(ds.source_id) == 8

    def test_row_limit_optional(self):
        ds = DataSource(uri="data/a.csv", row_limit=10_000)
        assert ds.row_limit == 10_000


# ─── SubTask ──────────────────────────────────────────────────────────────────


class TestSubTask:
    def test_basic_creation(self):
        st = SubTask(agent="etl", description="Load CSV")
        assert st.agent == "etl"
        assert st.required is True
        assert st.depends_on == []

    def test_auto_id_generated(self):
        st = SubTask(agent="etl", description="Load CSV")
        assert len(st.subtask_id) == 8


# ─── TaskPlan ─────────────────────────────────────────────────────────────────


class TestTaskPlan:
    def test_valid_plan(self, simple_task_plan):
        assert len(simple_task_plan.subtasks) == 3
        assert simple_task_plan.task_type == TaskType.FULL_PIPELINE

    def test_bad_dependency_raises(self, csv_source, task_id):
        bad_subtask = SubTask(
            subtask_id="st-001",
            agent="analysis",
            description="Analyse",
            depends_on=["st-DOES-NOT-EXIST"],
        )
        with pytest.raises(ValidationError, match="unknown id"):
            TaskPlan(
                task_id=task_id,
                raw_task="test",
                data_sources=[csv_source],
                task_type=TaskType.ANALYSIS_ONLY,
                subtasks=[bad_subtask],
            )

    def test_self_dependency_is_allowed_at_schema_level(self, csv_source, task_id):
        # The graph traversal handles cycle detection, not the schema
        st = SubTask(subtask_id="st-001", agent="etl", description="Load", depends_on=["st-001"])
        # Should not raise at schema level
        plan = TaskPlan(
            task_id=task_id,
            raw_task="test",
            data_sources=[csv_source],
            task_type=TaskType.ETL_ONLY,
            subtasks=[st],
        )
        assert plan is not None

    def test_subtasks_for_agent(self, simple_task_plan):
        etl_tasks = simple_task_plan.subtasks_for_agent("etl")
        assert len(etl_tasks) == 1
        assert etl_tasks[0].agent == "etl"

    def test_estimated_total_tokens(self, simple_task_plan):
        total = simple_task_plan.estimated_total_tokens()
        assert total == sum(st.estimated_tokens for st in simple_task_plan.subtasks)

    def test_empty_data_sources_allowed(self, task_id):
        """Analysis-only tasks may not need data sources (pre-loaded data)."""
        st = SubTask(subtask_id="st-001", agent="analysis", description="Analyse loaded data")
        plan = TaskPlan(
            task_id=task_id,
            raw_task="analyse loaded data",
            data_sources=[],
            task_type=TaskType.ANALYSIS_ONLY,
            subtasks=[st],
        )
        assert plan.data_sources == []


# ─── ETLResult ────────────────────────────────────────────────────────────────


class TestETLResult:
    def test_has_errors_true(self, task_id):
        result = ETLResult(
            task_id=task_id,
            source_ids=["s1"],
            row_count=0,
            column_count=2,
            schema=[],
            validation_issues=[ValidationIssue(severity="error", message="File not found")],
        )
        assert result.has_errors is True

    def test_has_errors_false_for_warnings(self, task_id):
        result = ETLResult(
            task_id=task_id,
            source_ids=["s1"],
            row_count=10,
            column_count=2,
            schema=[],
            validation_issues=[ValidationIssue(severity="warning", message="5% nulls")],
        )
        assert result.has_errors is False
        assert result.has_warnings is True

    def test_clean_result_has_no_issues(self, clean_etl_result):
        assert not clean_etl_result.has_errors
        assert not clean_etl_result.has_warnings


# ─── AnalysisResult ───────────────────────────────────────────────────────────


class TestAnalysisResult:
    def test_anomaly_count_property(self, clean_analysis_result):
        assert clean_analysis_result.anomaly_count == 1

    def test_anomaly_rate_bounds(self, task_id):
        with pytest.raises(ValidationError):
            AnalysisResult(task_id=task_id, anomaly_rate=1.5)

        with pytest.raises(ValidationError):
            AnalysisResult(task_id=task_id, anomaly_rate=-0.1)


# ─── CostSummary ──────────────────────────────────────────────────────────────


class TestCostSummary:
    def test_from_entries(self, task_id):
        entries = [
            CostEntry(
                task_id=task_id, agent_id="etl", model="claude-haiku-4-5-20251001",
                input_tokens=1000, output_tokens=500, cost_usd=0.001, latency_ms=300.0,
            ),
            CostEntry(
                task_id=task_id, agent_id="analysis", model="claude-haiku-4-5-20251001",
                input_tokens=800, output_tokens=400, cost_usd=0.0008, latency_ms=250.0,
            ),
            CostEntry(
                task_id=task_id, agent_id="etl", model="claude-sonnet-4-20250514",
                input_tokens=200, output_tokens=100, cost_usd=0.002, latency_ms=150.0,
            ),
        ]
        summary = CostSummary.from_entries(task_id=task_id, entries=entries)

        assert summary.total_cost_usd == pytest.approx(0.0038)
        assert summary.total_input_tokens == 2000
        assert summary.total_output_tokens == 1000
        assert summary.per_agent["etl"] == pytest.approx(0.003)
        assert summary.per_agent["analysis"] == pytest.approx(0.0008)
        assert len(summary.per_model) == 2

    def test_empty_entries(self, task_id):
        summary = CostSummary.from_entries(task_id=task_id, entries=[])
        assert summary.total_cost_usd == pytest.approx(0.0)
        assert summary.per_agent == {}
