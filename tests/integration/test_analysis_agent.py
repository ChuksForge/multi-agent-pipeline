"""
tests/integration/test_analysis_agent.py
──────────────────────────────────────────
Integration tests for analysis_node().

Tests the full agent function: PipelineState in → state update out.
Mocks only the LLM call (key_findings generation) — all tool logic is real.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl
import numpy as np

from pipeline.agents.analysis.agent import analysis_node
from pipeline.core.schemas import (
    AnalysisResult,
    ColumnSchema,
    DataSource,
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


def _make_etl_result(df: pl.DataFrame, task_id: str = "test-001") -> ETLResult:
    """Build an ETLResult from a polars DataFrame."""
    return ETLResult(
        task_id=task_id,
        source_ids=["src-001"],
        row_count=len(df),
        column_count=len(df.columns),
        schema=[
            ColumnSchema(name=c, dtype=str(df[c].dtype), nullable=False, null_rate=0.0)
            for c in df.columns
        ],
        data_json=df.write_json(),
    )


def _make_state(etl_result: ETLResult | None, task_id: str = "test-001") -> PipelineState:
    """Build a PipelineState with an optional ETLResult."""
    st = SubTask(subtask_id="st-analysis", agent="analysis", description="Analyse data")
    plan = TaskPlan(
        task_id=task_id,
        raw_task="Analyse data for anomalies",
        data_sources=[DataSource(uri="data/test.csv")],
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[st],
        output_format=OutputFormat.MARKDOWN,
        complexity=TaskComplexity.LOW,
    )
    state: PipelineState = initial_state(task_id=task_id, raw_task="test")  # type: ignore[assignment]
    state["task_plan"] = plan
    state["etl_result"] = etl_result
    state["status"] = PipelineStatus.RUNNING
    return state


@pytest.fixture
def clean_df():
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "revenue": rng.normal(loc=1000.0, scale=100.0, size=50).tolist(),
        "units": rng.integers(10, 100, size=50).tolist(),
        "cost": rng.normal(loc=400.0, scale=50.0, size=50).tolist(),
    })


@pytest.fixture
def anomalous_df():
    rng = np.random.default_rng(42)
    revenue = rng.normal(loc=1000.0, scale=50.0, size=47).tolist() + [9999.0, -500.0, 8000.0]
    units = rng.integers(10, 100, size=50).tolist()
    cost = rng.normal(loc=400.0, scale=30.0, size=50).tolist()
    return pl.DataFrame({"revenue": revenue, "units": units, "cost": cost})


@pytest.fixture
def tiny_df():
    return pl.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})


def _mock_llm_response(findings: list[str] | None = None):
    """Create a mock LLM response returning key findings JSON."""
    findings = findings or ["Test finding 1.", "Test finding 2."]
    mock_response = MagicMock()
    mock_response.content = json.dumps(findings)
    return mock_response


# ─── Happy path ───────────────────────────────────────────────────────────────


class TestAnalysisAgentHappyPath:
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_returns_dict(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert isinstance(update, dict)

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_analysis_result_in_update(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert "analysis_result" in update
        assert isinstance(update["analysis_result"], AnalysisResult)

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_summary_stats_populated(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        assert len(result.summary_stats) == 3  # revenue, units, cost

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_anomaly_rate_bounded(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        assert 0.0 <= result.anomaly_rate <= 1.0

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_anomalies_detected_in_anomalous_data(self, mock_llm_cls, anomalous_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(anomalous_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        assert result.anomaly_count >= 1

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_charts_generated(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        assert len(result.charts) >= 1

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_key_findings_from_llm(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response(
            ["Revenue is stable.", "No anomalies detected."]
        )
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        assert len(result.key_findings) >= 1

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_status_running_on_success(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert update["status"] == PipelineStatus.RUNNING

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_recovery_tier_none_on_clean_run(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert update["analysis_result"].recovery_tier == RecoveryTier.NONE

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_elapsed_seconds_positive(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert update["analysis_result"].elapsed_seconds > 0

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_current_agent_cleared(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        assert update["current_agent"] is None


# ─── No ETL data ──────────────────────────────────────────────────────────────


class TestAnalysisAgentNoData:
    def test_none_etl_result_produces_degraded(self):
        state = _make_state(etl_result=None)
        update = analysis_node(state)
        assert update["analysis_result"].recovery_tier == RecoveryTier.DEGRADED

    def test_none_etl_result_status_partial(self):
        state = _make_state(etl_result=None)
        update = analysis_node(state)
        assert update["status"] == PipelineStatus.PARTIAL

    def test_zero_row_etl_produces_degraded(self):
        empty_result = ETLResult(
            task_id="test-001", source_ids=[], row_count=0,
            column_count=0, schema=[],
        )
        state = _make_state(etl_result=empty_result)
        update = analysis_node(state)
        assert update["analysis_result"].recovery_tier == RecoveryTier.DEGRADED

    def test_none_etl_does_not_raise(self):
        state = _make_state(etl_result=None)
        try:
            update = analysis_node(state)
            assert isinstance(update, dict)
        except Exception as e:
            pytest.fail(f"analysis_node raised unexpectedly: {e}")


# ─── Insufficient data (tier 2) ───────────────────────────────────────────────


class TestAnalysisAgentInsufficientData:
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_tiny_df_falls_back_to_stats_only(self, mock_llm_cls, tiny_df):
        """2-row DataFrame triggers InsufficientDataError → stats-only fallback."""
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(tiny_df))
        update = analysis_node(state)
        result = update["analysis_result"]
        # Should still return a result, not crash
        assert isinstance(result, AnalysisResult)
        assert result.anomaly_count == 0

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_tiny_df_status_partial(self, mock_llm_cls, tiny_df):
        mock_llm_cls.return_value.invoke.return_value = _mock_llm_response()
        state = _make_state(_make_etl_result(tiny_df))
        update = analysis_node(state)
        assert update["status"] == PipelineStatus.PARTIAL


# ─── LLM failure fallback ─────────────────────────────────────────────────────


class TestAnalysisAgentLLMFailure:
    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_llm_failure_uses_fallback_findings(self, mock_llm_cls, clean_df):
        """LLM call fails — should use deterministic fallback findings."""
        mock_llm_cls.return_value.invoke.side_effect = RuntimeError("API timeout")
        state = _make_state(_make_etl_result(clean_df))
        update = analysis_node(state)
        result: AnalysisResult = update["analysis_result"]
        # Fallback findings should still be populated
        assert len(result.key_findings) >= 1

    @patch("pipeline.agents.analysis.agent.ChatAnthropic")
    def test_llm_failure_does_not_crash_agent(self, mock_llm_cls, clean_df):
        mock_llm_cls.return_value.invoke.side_effect = RuntimeError("rate limit")
        state = _make_state(_make_etl_result(clean_df))
        try:
            update = analysis_node(state)
            assert isinstance(update, dict)
        except Exception as e:
            pytest.fail(f"analysis_node raised on LLM failure: {e}")
