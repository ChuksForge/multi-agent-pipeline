"""
tests/integration/test_report_agent.py
─────────────────────────────────────────
Integration tests for report_node().
LLM call (executive summary) is mocked via @patch.
All formatter/emitter logic runs for real.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import numpy as np
import polars as pl

from pipeline.agents.report.agent import report_node
from pipeline.core.schemas import (
    AnalysisResult,
    AnomalyRecord,
    ColumnSchema,
    DataSource,
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


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_plan(output_format: OutputFormat = OutputFormat.MARKDOWN,
               task_id: str = "rpt-001") -> TaskPlan:
    return TaskPlan(
        task_id=task_id,
        raw_task="Analyse sales data and report anomalies",
        data_sources=[DataSource(uri="data/sales.csv")],
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[SubTask(subtask_id="st-1", agent="report", description="Generate report")],
        output_format=output_format,
        complexity=TaskComplexity.LOW,
    )


def _make_etl(task_id: str = "rpt-001") -> ETLResult:
    return ETLResult(
        task_id=task_id,
        source_ids=["src-001"],
        row_count=500,
        column_count=3,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
            ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
            ColumnSchema(name="units", dtype="Int64", nullable=False, null_rate=0.0),
        ],
        elapsed_seconds=1.1,
    )


def _make_analysis(task_id: str = "rpt-001") -> AnalysisResult:
    return AnalysisResult(
        task_id=task_id,
        summary_stats=[
            SummaryStats(column="revenue", dtype="Float64", count=500, null_count=0,
                         mean=1200.0, std=300.0, min=100.0, max=9999.0,
                         p25=900.0, p50=1200.0, p75=1500.0),
        ],
        anomalies=[
            AnomalyRecord(row_index=42, column="revenue", value=9999.0,
                          anomaly_score=-0.5, method="ensemble"),
        ],
        anomaly_rate=0.002,
        key_findings=["Revenue has one significant outlier at row 42."],
    )


def _make_state(
    plan: TaskPlan | None,
    etl: ETLResult | None,
    analysis: AnalysisResult | None,
    task_id: str = "rpt-001",
) -> PipelineState:
    state: PipelineState = initial_state(task_id=task_id, raw_task="test")  # type: ignore[assignment]
    state["task_plan"] = plan
    state["etl_result"] = etl
    state["analysis_result"] = analysis
    state["status"] = PipelineStatus.RUNNING
    return state


def _mock_llm(summary: str = "This is the executive summary."):
    mock = MagicMock()
    mock.content = summary
    return mock


# ─── Happy path — Markdown ────────────────────────────────────────────────────


class TestReportAgentMarkdown:
    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_returns_dict(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert isinstance(update, dict)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_report_result_in_update(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert "report_result" in update
        assert isinstance(update["report_result"], ReportResult)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_output_format_markdown(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(OutputFormat.MARKDOWN), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].output_format == OutputFormat.MARKDOWN

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_full_content_is_markdown(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        content = update["report_result"].full_content
        assert "# Data Pipeline Report" in content
        assert "## Anomaly Detection" in content

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_executive_summary_in_report(self, mock_cls):
        summary = "This is the mocked executive summary for testing."
        mock_cls.return_value.invoke.return_value = _mock_llm(summary)
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert summary in update["report_result"].full_content

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_word_count_positive(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].word_count > 50

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_status_complete_on_clean_run(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["status"] == PipelineStatus.COMPLETE

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_recovery_tier_none_on_clean_run(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].recovery_tier == RecoveryTier.NONE

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_current_agent_cleared(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["current_agent"] is None

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_sections_populated(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert len(update["report_result"].sections) >= 3

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_elapsed_seconds_positive(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].elapsed_seconds > 0


# ─── JSON output format ───────────────────────────────────────────────────────


class TestReportAgentJSON:
    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_json_format_produces_valid_json(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(OutputFormat.JSON), _make_etl(), _make_analysis())
        update = report_node(state)
        content = update["report_result"].full_content
        parsed = json.loads(content)
        assert "meta" in parsed
        assert "etl" in parsed
        assert "analysis" in parsed

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_json_output_format_set(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(OutputFormat.JSON), _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].output_format == OutputFormat.JSON


# ─── Missing data recovery ────────────────────────────────────────────────────


class TestReportAgentRecovery:
    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_no_task_plan_produces_degraded(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(None, _make_etl(), _make_analysis())
        update = report_node(state)
        assert update["report_result"].recovery_tier == RecoveryTier.DEGRADED

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_no_task_plan_does_not_raise(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(None, _make_etl(), _make_analysis())
        try:
            update = report_node(state)
            assert isinstance(update, dict)
        except Exception as e:
            pytest.fail(f"report_node raised unexpectedly: {e}")

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_no_etl_result_still_produces_report(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), None, _make_analysis())
        update = report_node(state)
        assert isinstance(update["report_result"], ReportResult)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_no_analysis_result_still_produces_report(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm()
        state = _make_state(_make_plan(), _make_etl(), None)
        update = report_node(state)
        assert isinstance(update["report_result"], ReportResult)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_llm_failure_uses_fallback_summary(self, mock_cls):
        mock_cls.return_value.invoke.side_effect = RuntimeError("API down")
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        update = report_node(state)
        # Should still produce a report using fallback summary
        result = update["report_result"]
        assert isinstance(result, ReportResult)
        assert len(result.full_content) > 100

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    def test_llm_failure_does_not_crash(self, mock_cls):
        mock_cls.return_value.invoke.side_effect = RuntimeError("timeout")
        state = _make_state(_make_plan(), _make_etl(), _make_analysis())
        try:
            update = report_node(state)
            assert isinstance(update, dict)
        except Exception as e:
            pytest.fail(f"report_node raised on LLM failure: {e}")
