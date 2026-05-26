"""
tests/unit/test_eval_metrics.py
─────────────────────────────────
Unit tests for eval metrics: task_completion_rate, output_correctness_score,
cost_per_run, aggregate_results. No pipeline runs — pure unit tests on typed inputs.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from pipeline.core.schemas import (
    AnalysisResult,
    AnomalyRecord,
    ColumnSchema,
    CostEntry,
    ETLResult,
    OutputFormat,
    PipelineStatus,
    RecoveryTier,
    ReportResult,
    SummaryStats,
    ValidationIssue,
)
from evals.metrics import (
    ExpectedOutput,
    EvalSummary,
    CorrectnessResult,
    aggregate_results,
    cost_per_run,
    output_correctness_score,
    task_completion_rate,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_etl():
    return ETLResult(
        task_id="t1", source_ids=["s1"], row_count=500, column_count=4,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
            ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
            ColumnSchema(name="units", dtype="Int64", nullable=False, null_rate=0.0),
            ColumnSchema(name="region", dtype="Utf8", nullable=False, null_rate=0.0),
        ],
    )


@pytest.fixture
def clean_analysis():
    return AnalysisResult(
        task_id="t1",
        summary_stats=[
            SummaryStats(column="revenue", dtype="Float64", count=500, null_count=0,
                         mean=1200.0, std=200.0, min=100.0, max=9999.0,
                         p25=900.0, p50=1200.0, p75=1500.0),
        ],
        anomalies=[
            AnomalyRecord(row_index=42, column="revenue", value=9999.0,
                          anomaly_score=-0.5, method="ensemble"),
        ],
        anomaly_rate=0.002,
        key_findings=["One outlier found."],
    )


@pytest.fixture
def clean_report():
    return ReportResult(
        task_id="t1", output_format=OutputFormat.MARKDOWN,
        title="Test Report",
        full_content=(
            "# Data Pipeline Report\n\n"
            "## Executive Summary\n\nGood data.\n\n"
            "## Anomaly Detection\n\n1 anomaly found.\n\n"
            "## Statistical Summary\n\nMean revenue: 1200.\n"
        ),
        word_count=25,
    )


@pytest.fixture
def default_expected():
    return ExpectedOutput()


@pytest.fixture
def strict_expected():
    return ExpectedOutput(
        status="complete",
        min_rows=100,
        required_columns=["date", "revenue", "units", "region"],
        max_anomaly_rate=0.05,
        min_anomaly_count=1,
        required_report_sections=["Executive Summary", "Anomaly Detection"],
        min_word_count=20,
    )


# ─── task_completion_rate ─────────────────────────────────────────────────────


class TestTaskCompletionRate:
    def test_all_complete(self):
        statuses = [PipelineStatus.COMPLETE] * 5
        assert task_completion_rate(statuses) == pytest.approx(1.0)

    def test_all_failed(self):
        statuses = [PipelineStatus.FAILED] * 5
        assert task_completion_rate(statuses) == pytest.approx(0.0)

    def test_partial_counts_as_complete(self):
        statuses = [PipelineStatus.COMPLETE, PipelineStatus.PARTIAL, PipelineStatus.FAILED]
        rate = task_completion_rate(statuses)
        assert rate == pytest.approx(2 / 3, rel=1e-4)

    def test_empty_list_returns_zero(self):
        assert task_completion_rate([]) == pytest.approx(0.0)

    def test_mixed_statuses(self):
        statuses = [
            PipelineStatus.COMPLETE,
            PipelineStatus.COMPLETE,
            PipelineStatus.PARTIAL,
            PipelineStatus.FAILED,
        ]
        assert task_completion_rate(statuses) == pytest.approx(0.75)

    def test_result_bounded_between_zero_and_one(self):
        import random
        statuses = [random.choice(list(PipelineStatus)) for _ in range(20)]
        rate = task_completion_rate(statuses)
        assert 0.0 <= rate <= 1.0


# ─── output_correctness_score ─────────────────────────────────────────────────


class TestOutputCorrectnessScore:
    def test_returns_correctness_result(self, clean_etl, clean_analysis,
                                        clean_report, default_expected):
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, clean_report, default_expected
        )
        assert isinstance(result, CorrectnessResult)

    def test_perfect_score_on_clean_data_with_loose_expected(
            self, clean_etl, clean_analysis, clean_report, default_expected):
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, clean_report, default_expected
        )
        assert result.total_score == pytest.approx(1.0)

    def test_passes_with_clean_data_strict_expected(
            self, clean_etl, clean_analysis, clean_report, strict_expected):
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, clean_report, strict_expected
        )
        assert result.passed is True
        assert result.total_score >= 0.60

    def test_none_etl_penalises_schema_score(self, clean_analysis,
                                              clean_report, default_expected):
        result = output_correctness_score(
            "t1", None, clean_analysis, clean_report, default_expected
        )
        assert result.schema_score == pytest.approx(0.0)
        assert "ETL result missing" in result.failures

    def test_none_analysis_penalises_analysis_score(self, clean_etl,
                                                      clean_report, default_expected):
        result = output_correctness_score(
            "t1", clean_etl, None, clean_report, default_expected
        )
        assert result.analysis_score == pytest.approx(0.0)

    def test_none_report_penalises_report_score(self, clean_etl,
                                                 clean_analysis, default_expected):
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, None, default_expected
        )
        assert result.report_score == pytest.approx(0.0)

    def test_missing_required_columns_reduces_schema_score(
            self, clean_analysis, clean_report):
        etl_missing_cols = ETLResult(
            task_id="t1", source_ids=["s1"], row_count=500, column_count=2,
            schema=[
                ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
                ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
            ],
        )
        expected = ExpectedOutput(required_columns=["date", "revenue", "units", "region"])
        result = output_correctness_score(
            "t1", etl_missing_cols, clean_analysis, clean_report, expected
        )
        assert result.schema_score < 1.0
        assert len(result.failures) > 0

    def test_row_count_below_minimum_penalises_schema(
            self, clean_analysis, clean_report):
        small_etl = ETLResult(
            task_id="t1", source_ids=["s1"], row_count=5, column_count=2, schema=[],
        )
        expected = ExpectedOutput(min_rows=100)
        result = output_correctness_score(
            "t1", small_etl, clean_analysis, clean_report, expected
        )
        assert result.schema_score < 1.0

    def test_anomaly_rate_above_max_penalises_analysis(
            self, clean_etl, clean_report):
        high_anomaly = AnalysisResult(
            task_id="t1",
            anomaly_rate=0.50,  # 50% anomalies
            anomalies=[AnomalyRecord(row_index=i, value=i, anomaly_score=-0.5,
                                      method="zscore") for i in range(250)],
            summary_stats=[SummaryStats(column="x", dtype="Float64",
                                         count=500, null_count=0)],
        )
        expected = ExpectedOutput(max_anomaly_rate=0.05)
        result = output_correctness_score(
            "t1", clean_etl, high_anomaly, clean_report, expected
        )
        assert result.analysis_score < 1.0

    def test_missing_report_section_penalises_report_score(
            self, clean_etl, clean_analysis):
        short_report = ReportResult(
            task_id="t1", output_format=OutputFormat.MARKDOWN,
            title="Test", full_content="# Report\n\nHello world.",
            word_count=10,
        )
        expected = ExpectedOutput(required_report_sections=["Anomaly Detection"])
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, short_report, expected
        )
        assert result.report_score < 1.0

    def test_score_bounded_between_zero_and_one(self, clean_etl,
                                                 clean_analysis, clean_report):
        expected = ExpectedOutput(
            min_rows=10000, required_columns=["nonexistent"],
            min_word_count=99999,
        )
        result = output_correctness_score(
            "t1", clean_etl, clean_analysis, clean_report, expected
        )
        assert 0.0 <= result.total_score <= 1.0

    def test_all_none_returns_zero_score(self, default_expected):
        result = output_correctness_score("t1", None, None, None, default_expected)
        assert result.total_score == pytest.approx(0.0)
        assert result.passed is False


# ─── cost_per_run ─────────────────────────────────────────────────────────────


class TestCostPerRun:
    def test_empty_entries_returns_zero_cost(self):
        summary = cost_per_run("t1", [])
        assert summary.total_cost_usd == pytest.approx(0.0)

    def test_single_entry(self):
        entries = [
            CostEntry(task_id="t1", agent_id="etl",
                      model="claude-haiku-4-5-20251001",
                      input_tokens=500, output_tokens=300, cost_usd=0.001,
                      latency_ms=400.0),
        ]
        summary = cost_per_run("t1", entries)
        assert summary.total_cost_usd == pytest.approx(0.001)
        assert summary.total_input_tokens == 500
        assert summary.total_output_tokens == 300

    def test_multiple_agents_summed(self):
        entries = [
            CostEntry(task_id="t1", agent_id="etl",
                      model="claude-haiku-4-5-20251001",
                      input_tokens=400, output_tokens=200, cost_usd=0.0005,
                      latency_ms=300.0),
            CostEntry(task_id="t1", agent_id="analysis",
                      model="claude-haiku-4-5-20251001",
                      input_tokens=600, output_tokens=400, cost_usd=0.0008,
                      latency_ms=500.0),
        ]
        summary = cost_per_run("t1", entries)
        assert summary.total_cost_usd == pytest.approx(0.0013)
        assert "etl" in summary.per_agent
        assert "analysis" in summary.per_agent


# ─── aggregate_results ────────────────────────────────────────────────────────


class TestAggregateResults:
    def _make_result(self, status="complete", score=0.8, cost=0.001,
                     latency=500.0, passed=True):
        return {
            "task_id": "t1", "status": status,
            "correctness_score": score, "cost_usd": cost,
            "latency_ms": latency, "passed": passed,
        }

    def test_empty_returns_zero_summary(self):
        summary = aggregate_results([])
        assert summary.total_tasks == 0
        assert summary.completion_rate == pytest.approx(0.0)

    def test_total_tasks_count(self):
        results = [self._make_result() for _ in range(5)]
        summary = aggregate_results(results)
        assert summary.total_tasks == 5

    def test_completion_rate_correct(self):
        results = [
            self._make_result(status="complete"),
            self._make_result(status="partial"),
            self._make_result(status="failed", passed=False),
        ]
        summary = aggregate_results(results)
        assert summary.completion_rate == pytest.approx(2 / 3, rel=1e-4)

    def test_avg_correctness_correct(self):
        results = [
            self._make_result(score=1.0),
            self._make_result(score=0.5),
        ]
        summary = aggregate_results(results)
        assert summary.avg_correctness == pytest.approx(0.75)

    def test_avg_cost_correct(self):
        results = [
            self._make_result(cost=0.002),
            self._make_result(cost=0.004),
        ]
        summary = aggregate_results(results)
        assert summary.avg_cost_usd == pytest.approx(0.003)

    def test_passed_count_correct(self):
        results = [
            self._make_result(passed=True),
            self._make_result(passed=True),
            self._make_result(passed=False),
        ]
        summary = aggregate_results(results)
        assert summary.passed_count == 2

    def test_failed_count_correct(self):
        results = [
            self._make_result(status="complete"),
            self._make_result(status="failed", passed=False),
            self._make_result(status="failed", passed=False),
        ]
        summary = aggregate_results(results)
        assert summary.failed_count == 2

    def test_partial_count_correct(self):
        results = [
            self._make_result(status="partial"),
            self._make_result(status="complete"),
        ]
        summary = aggregate_results(results)
        assert summary.partial_count == 1

    def test_per_task_preserved(self):
        results = [self._make_result() for _ in range(3)]
        summary = aggregate_results(results)
        assert len(summary.per_task) == 3
