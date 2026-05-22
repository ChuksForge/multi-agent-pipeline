"""
tests/conftest.py
─────────────────
Shared pytest fixtures used across unit and integration tests.
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import MagicMock

import pytest

# Ensure test env vars are set before any settings import
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key-for-unit-tests")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key-for-unit-tests")


from pipeline.core.schemas import (
    AnalysisResult,
    AnomalyRecord,
    ChartSpec,
    ColumnSchema,
    CostEntry,
    DataSource,
    DataSourceType,
    ETLResult,
    OutputFormat,
    PipelineStatus,
    RecoveryTier,
    ReportResult,
    SubTask,
    SummaryStats,
    TaskComplexity,
    TaskPlan,
    TaskType,
    ValidationIssue,
)
from pipeline.core.state import PipelineState, initial_state
from pipeline.middleware.token_tracker import TokenTracker


# ─── IDs ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def task_id() -> str:
    return str(uuid.uuid4())


# ─── DataSource fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def csv_source(tmp_path) -> DataSource:
    """A DataSource pointing to a real temp CSV file."""
    p = tmp_path / "sales.csv"
    p.write_text("date,revenue,units\n2024-01-01,1000,10\n2024-01-02,1200,12\n")
    return DataSource(uri=str(p), source_type=DataSourceType.CSV)


@pytest.fixture
def parquet_source() -> DataSource:
    return DataSource(uri="s3://bucket/data.parquet", source_type=DataSourceType.PARQUET)


# ─── TaskPlan fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def simple_task_plan(csv_source, task_id) -> TaskPlan:
    etl_subtask = SubTask(subtask_id="st-001", agent="etl", description="Load and validate sales CSV")
    analysis_subtask = SubTask(
        subtask_id="st-002",
        agent="analysis",
        description="Detect anomalies in revenue column",
        depends_on=["st-001"],
    )
    report_subtask = SubTask(
        subtask_id="st-003",
        agent="report",
        description="Generate markdown summary report",
        depends_on=["st-002"],
    )
    return TaskPlan(
        task_id=task_id,
        raw_task="Analyse sales data and flag anomalies, output a markdown report",
        data_sources=[csv_source],
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[etl_subtask, analysis_subtask, report_subtask],
        output_format=OutputFormat.MARKDOWN,
        complexity=TaskComplexity.LOW,
    )


# ─── ETLResult fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def clean_etl_result(task_id) -> ETLResult:
    return ETLResult(
        task_id=task_id,
        source_ids=["src-001"],
        row_count=100,
        column_count=3,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
            ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
            ColumnSchema(name="units", dtype="Int64", nullable=False, null_rate=0.0),
        ],
    )


@pytest.fixture
def etl_result_with_warnings(task_id) -> ETLResult:
    return ETLResult(
        task_id=task_id,
        source_ids=["src-001"],
        row_count=100,
        column_count=3,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=True, null_rate=0.05),
            ColumnSchema(name="revenue", dtype="Float64", nullable=True, null_rate=0.0),
            ColumnSchema(name="units", dtype="Int64", nullable=False, null_rate=0.0),
        ],
        validation_issues=[
            ValidationIssue(column="date", severity="warning", message="5% null values", affected_rows=5),
        ],
    )


# ─── AnalysisResult fixtures ──────────────────────────────────────────────────


@pytest.fixture
def clean_analysis_result(task_id) -> AnalysisResult:
    return AnalysisResult(
        task_id=task_id,
        summary_stats=[
            SummaryStats(
                column="revenue", dtype="Float64", count=100, null_count=0,
                mean=1100.0, std=200.0, min=500.0, max=2000.0,
                p25=950.0, p50=1100.0, p75=1300.0,
            )
        ],
        anomalies=[
            AnomalyRecord(row_index=42, column="revenue", value=9999.0, anomaly_score=-0.45, method="isolation_forest"),
        ],
        anomaly_rate=0.01,
        key_findings=["1 anomaly detected in revenue column on row 42"],
    )


# ─── PipelineState fixtures ───────────────────────────────────────────────────


@pytest.fixture
def empty_state(task_id) -> PipelineState:
    return initial_state(task_id=task_id, raw_task="test task")


@pytest.fixture
def state_after_etl(empty_state, simple_task_plan, clean_etl_result) -> PipelineState:
    # TypedDict does not support **kwargs construction — mutate the dict directly
    state: PipelineState = dict(empty_state)  # type: ignore[assignment]
    state["task_plan"] = simple_task_plan
    state["etl_result"] = clean_etl_result
    state["status"] = PipelineStatus.RUNNING
    return state


# ─── TokenTracker fixtures ────────────────────────────────────────────────────


@pytest.fixture
def token_tracker(task_id) -> TokenTracker:
    return TokenTracker(task_id=task_id, agent_id="etl")


@pytest.fixture
def populated_tracker(task_id) -> TokenTracker:
    """A tracker with pre-populated cost entries for aggregation tests."""
    tracker = TokenTracker(task_id=task_id, agent_id="etl")
    tracker.entries = [
        CostEntry(
            task_id=task_id, agent_id="etl",
            model="claude-haiku-4-5-20251001",
            input_tokens=500, output_tokens=300,
            cost_usd=0.0001616, latency_ms=450.0,
        ),
        CostEntry(
            task_id=task_id, agent_id="etl",
            model="claude-haiku-4-5-20251001",
            input_tokens=200, output_tokens=150,
            cost_usd=0.000076, latency_ms=220.0,
        ),
    ]
    return tracker
