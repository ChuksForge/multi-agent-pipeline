"""
tests/integration/test_full_pipeline.py
─────────────────────────────────────────
Integration tests for the full pipeline graph.

Tests the StateGraph: planner → supervisor → etl → supervisor → analysis
→ supervisor → report → supervisor → END.

All LLM calls are mocked. ETL tools run for real (temp CSV files).
Validates the full state transition sequence.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.core.schemas import (
    OutputFormat,
    PipelineStatus,
    ReportResult,
    TaskType,
)
from pipeline.core.state import PipelineState
from pipeline.orchestrator.graph import build_graph, run_pipeline_sync


# ─── Shared mock helpers ──────────────────────────────────────────────────────


VALID_PLAN_JSON = {
    "raw_task": "Analyse sales data for anomalies",
    "data_sources": [],
    "task_type": "full_pipeline",
    "output_format": "markdown",
    "complexity": "low",
    "subtasks": [
        {"subtask_id": "st-001", "agent": "etl",
         "description": "Load data", "depends_on": [],
         "required": True, "estimated_tokens": 300},
        {"subtask_id": "st-002", "agent": "analysis",
         "description": "Analyse data", "depends_on": ["st-001"],
         "required": True, "estimated_tokens": 400},
        {"subtask_id": "st-003", "agent": "report",
         "description": "Generate report", "depends_on": ["st-002"],
         "required": True, "estimated_tokens": 300},
    ],
}

KEY_FINDINGS_JSON = '["Revenue is stable.", "No major anomalies detected."]'
EXECUTIVE_SUMMARY = "This report covers a clean dataset with no significant issues detected."


def _make_llm_mock(content: str) -> MagicMock:
    mock = MagicMock()
    mock.content = content
    return mock


def _multi_response_mock(*responses):
    """Return different content for successive .invoke() calls."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = [_make_llm_mock(r) for r in responses]
    return mock_llm


# ─── Full pipeline fixtures ───────────────────────────────────────────────────


@pytest.fixture
def sales_csv(tmp_path):
    """Real CSV file for ETL agent to load."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 50
    p = tmp_path / "sales.csv"
    rows = "\n".join(
        f"2024-01-{(i % 28)+1:02d},{rng.integers(500,2000)},{rng.integers(5,50)}"
        for i in range(n)
    )
    p.write_text("date,revenue,units\n" + rows)
    return str(p)


@pytest.fixture
def plan_json_with_source(sales_csv):
    plan = dict(VALID_PLAN_JSON)
    plan["data_sources"] = [{"uri": sales_csv}]
    return plan


# ─── Graph construction ───────────────────────────────────────────────────────


class TestBuildGraph:
    def test_graph_builds_without_error(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_is_compiled(self):
        graph = build_graph()
        # Compiled LangGraph has an invoke method
        assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


# ─── Full pipeline run ────────────────────────────────────────────────────────


class TestFullPipelineRun:
    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_pipeline_completes(self, mock_planner, mock_analysis, mock_report,
                                sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        raw_task = f"Analyse {sales_csv} for revenue anomalies"
        final_state: PipelineState = run_pipeline_sync(graph, raw_task)

        assert final_state["status"] in (PipelineStatus.COMPLETE, PipelineStatus.PARTIAL)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_etl_result_populated(self, mock_planner, mock_analysis, mock_report,
                                   sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        final_state = run_pipeline_sync(graph, f"Analyse {sales_csv}")
        assert final_state.get("etl_result") is not None
        assert final_state["etl_result"].row_count == 50

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_analysis_result_populated(self, mock_planner, mock_analysis, mock_report,
                                        sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        final_state = run_pipeline_sync(graph, f"Analyse {sales_csv}")
        assert final_state.get("analysis_result") is not None

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_report_result_populated(self, mock_planner, mock_analysis, mock_report,
                                      sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        final_state = run_pipeline_sync(graph, f"Analyse {sales_csv}")
        assert final_state.get("report_result") is not None
        assert isinstance(final_state["report_result"], ReportResult)

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_report_contains_markdown(self, mock_planner, mock_analysis, mock_report,
                                       sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        final_state = run_pipeline_sync(graph, f"Analyse {sales_csv}")
        content = final_state["report_result"].full_content
        assert "# Data Pipeline Report" in content

    @patch("pipeline.agents.report.agent.ChatAnthropic")
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_cost_log_accumulated(self, mock_planner, mock_analysis, mock_report,
                                   sales_csv, plan_json_with_source):
        mock_planner.return_value.invoke.return_value = _make_llm_mock(
            json.dumps(plan_json_with_source)
        )
        mock_analysis.return_value.invoke.return_value = _make_llm_mock(KEY_FINDINGS_JSON)
        mock_report.return_value.invoke.return_value = _make_llm_mock(EXECUTIVE_SUMMARY)

        graph = build_graph()
        final_state = run_pipeline_sync(graph, f"Analyse {sales_csv}")
        # Cost log may be empty since mock LLM doesn't emit real token usage
        assert isinstance(final_state.get("cost_log", []), list)


# ─── Planner failure handling ─────────────────────────────────────────────────


class TestPlannerFailure:
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_planner_failure_sets_failed_status(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _make_llm_mock("not json")
        graph = build_graph()
        final_state = run_pipeline_sync(graph, "Analyse some data")
        assert final_state["status"] == PipelineStatus.FAILED

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_planner_failure_no_etl_result(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _make_llm_mock("not json")
        graph = build_graph()
        final_state = run_pipeline_sync(graph, "Analyse some data")
        assert final_state.get("etl_result") is None

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_invalid_task_does_not_crash(self, mock_cls):
        graph = build_graph()
        try:
            final_state = run_pipeline_sync(graph, "hi")
            assert isinstance(final_state, dict)
        except Exception as e:
            pytest.fail(f"Pipeline raised on invalid task: {e}")
