"""
tests/unit/test_md_formatter.py
─────────────────────────────────
Unit tests for the Markdown report formatter.
Validates section presence, table structure, and content accuracy.
No LLM calls — all inputs are typed fixtures.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.report.tools.md_formatter import format_markdown, word_count
from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    AnomalyRecord,
    ColumnSchema,
    CostEntry,
    CostSummary,
    DataSource,
    ETLResult,
    OutputFormat,
    RecoveryTier,
    ReportSection,
    SubTask,
    SummaryStats,
    TaskComplexity,
    TaskPlan,
    TaskType,
    ValidationIssue,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def task_plan():
    return TaskPlan(
        task_id="test-report-001",
        raw_task="Analyse monthly sales data for anomalies",
        data_sources=[DataSource(uri="data/sales.csv")],
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[SubTask(subtask_id="st-1", agent="etl", description="Load data")],
        output_format=OutputFormat.MARKDOWN,
        complexity=TaskComplexity.LOW,
    )


@pytest.fixture
def clean_etl():
    return ETLResult(
        task_id="test-report-001",
        source_ids=["src-001"],
        row_count=1000,
        column_count=4,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
            ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
            ColumnSchema(name="units", dtype="Int64", nullable=False, null_rate=0.0),
            ColumnSchema(name="region", dtype="Utf8", nullable=True, null_rate=0.02),
        ],
        elapsed_seconds=1.23,
    )


@pytest.fixture
def etl_with_issues():
    return ETLResult(
        task_id="test-report-001",
        source_ids=["src-001"],
        row_count=800,
        column_count=4,
        schema=[
            ColumnSchema(name="revenue", dtype="Float64", nullable=True, null_rate=0.15),
        ],
        validation_issues=[
            ValidationIssue(column="revenue", severity="warning",
                            message="15% null values", affected_rows=120),
            ValidationIssue(column="id", severity="error",
                            message="Required column missing", affected_rows=None),
        ],
        elapsed_seconds=0.89,
    )


@pytest.fixture
def clean_analysis():
    return AnalysisResult(
        task_id="test-report-001",
        summary_stats=[
            SummaryStats(
                column="revenue", dtype="Float64", count=1000, null_count=0,
                mean=1250.0, std=320.0, min=100.0, max=9999.0,
                p25=950.0, p50=1200.0, p75=1550.0,
            ),
            SummaryStats(
                column="units", dtype="Int64", count=1000, null_count=0,
                mean=25.3, std=8.1, min=1.0, max=100.0,
                p25=18.0, p50=25.0, p75=32.0,
            ),
        ],
        anomalies=[
            AnomalyRecord(row_index=42, column="revenue", value=9999.0,
                          anomaly_score=-0.45, method="ensemble"),
            AnomalyRecord(row_index=187, column="revenue", value=9800.0,
                          anomaly_score=-0.41, method="isolation_forest"),
        ],
        anomaly_rate=0.002,
        key_findings=[
            "Revenue shows a strong upward trend over the period.",
            "2 anomalies detected — rows 42 and 187 have unusually high revenue.",
            "All other columns are within expected ranges.",
        ],
    )


@pytest.fixture
def cost_summary():
    entries = [
        CostEntry(task_id="test-report-001", agent_id="etl",
                  model="claude-haiku-4-5-20251001",
                  input_tokens=500, output_tokens=300, cost_usd=0.00052, latency_ms=450.0),
        CostEntry(task_id="test-report-001", agent_id="analysis",
                  model="claude-haiku-4-5-20251001",
                  input_tokens=800, output_tokens=400, cost_usd=0.00084, latency_ms=680.0),
        CostEntry(task_id="test-report-001", agent_id="report",
                  model="claude-haiku-4-5-20251001",
                  input_tokens=300, output_tokens=200, cost_usd=0.00032, latency_ms=320.0),
    ]
    return CostSummary.from_entries(task_id="test-report-001", entries=entries)


EXEC_SUMMARY = (
    "This report analyses 1,000 rows of monthly sales data across 4 columns. "
    "Two anomalies were detected in the revenue column using ensemble methods. "
    "Data quality is excellent with no errors found. "
    "The pipeline completed successfully."
)


# ─── Section presence ─────────────────────────────────────────────────────────


class TestMarkdownSectionPresence:
    def test_title_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "# Data Pipeline Report" in md

    def test_executive_summary_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Executive Summary" in md
        assert EXEC_SUMMARY[:30] in md

    def test_data_overview_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Data Overview" in md

    def test_schema_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Schema" in md

    def test_anomaly_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Anomaly Detection" in md

    def test_key_findings_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Key Findings" in md

    def test_stats_section_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Statistical Summary" in md

    def test_data_quality_absent_when_no_issues(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Data Quality" not in md

    def test_data_quality_present_when_issues_exist(self, task_plan, etl_with_issues, clean_analysis):
        md = format_markdown(task_plan, etl_with_issues, clean_analysis, EXEC_SUMMARY)
        assert "## Data Quality" in md

    def test_cost_section_present_when_provided(self, task_plan, clean_etl, clean_analysis, cost_summary):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY,
                             cost_summary=cost_summary)
        assert "## Cost Summary" in md

    def test_cost_section_absent_when_not_provided(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "## Cost Summary" not in md

    def test_errors_section_present_when_errors(self, task_plan, clean_etl, clean_analysis):
        errors = [AgentErrorRecord(
            agent_id="etl", attempt=1, error_type="TimeoutError",
            message="Agent timed out", recovery_tier=RecoveryTier.SIMPLIFIED,
        )]
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY,
                             errors=errors)
        assert "## Pipeline Errors" in md


# ─── Data accuracy ────────────────────────────────────────────────────────────


class TestMarkdownDataAccuracy:
    def test_row_count_in_overview(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "1,000" in md  # 1000 rows formatted with comma

    def test_column_names_in_schema(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "`date`" in md
        assert "`revenue`" in md
        assert "`units`" in md
        assert "`region`" in md

    def test_anomaly_count_in_report(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "2" in md  # 2 anomalies

    def test_anomaly_row_indices_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "42" in md
        assert "187" in md

    def test_key_findings_text_present(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "upward trend" in md

    def test_revenue_mean_in_stats(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "1250.00" in md

    def test_task_id_in_header(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert "test-report-001" in md

    def test_validation_error_shown(self, task_plan, etl_with_issues, clean_analysis):
        md = format_markdown(task_plan, etl_with_issues, clean_analysis, EXEC_SUMMARY)
        assert "Required column missing" in md

    def test_cost_totals_in_report(self, task_plan, clean_etl, clean_analysis, cost_summary):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY,
                             cost_summary=cost_summary)
        # Cost summary should include per-agent breakdown
        assert "`etl`" in md
        assert "`analysis`" in md
        assert "`report`" in md


# ─── No-anomaly case ──────────────────────────────────────────────────────────


class TestMarkdownNoAnomalies:
    def test_no_anomalies_message(self, task_plan, clean_etl):
        analysis_no_anomalies = AnalysisResult(
            task_id="test-report-001",
            anomaly_rate=0.0,
        )
        md = format_markdown(task_plan, clean_etl, analysis_no_anomalies, EXEC_SUMMARY)
        assert "No anomalies detected" in md

    def test_no_findings_section_when_empty(self, task_plan, clean_etl):
        analysis_empty = AnalysisResult(task_id="test-report-001")
        md = format_markdown(task_plan, clean_etl, analysis_empty, EXEC_SUMMARY)
        assert "## Key Findings" not in md


# ─── word_count utility ───────────────────────────────────────────────────────


class TestWordCount:
    def test_returns_integer(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert isinstance(word_count(md), int)

    def test_word_count_positive(self, task_plan, clean_etl, clean_analysis):
        md = format_markdown(task_plan, clean_etl, clean_analysis, EXEC_SUMMARY)
        assert word_count(md) > 50

    def test_empty_string_word_count(self):
        assert word_count("") == 0

    def test_code_blocks_excluded(self):
        md = "Hello world\n```json\n{\"key\": \"value\"}\n```"
        # "Hello world" = 2 words; code block should be stripped
        count = word_count(md)
        assert count < 10  # far less than if code block were counted
