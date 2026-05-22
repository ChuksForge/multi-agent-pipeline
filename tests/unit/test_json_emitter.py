"""
tests/unit/test_json_emitter.py
─────────────────────────────────
Unit tests for JSON report emitter.
Validates JSON structure, schema completeness, and serialisation correctness.
"""

from __future__ import annotations

import json
import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from pipeline.agents.report.tools.json_emitter import emit_json
from pipeline.agents.report.tools.pdf_renderer import is_available as pdf_available
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
    SubTask,
    SummaryStats,
    TaskComplexity,
    TaskPlan,
    TaskType,
    ValidationIssue,
)

EXEC_SUMMARY = "Test executive summary paragraph for unit testing."


@pytest.fixture
def task_plan():
    return TaskPlan(
        task_id="json-test-001",
        raw_task="Test task for JSON emission",
        data_sources=[DataSource(uri="data/test.csv")],
        task_type=TaskType.FULL_PIPELINE,
        subtasks=[SubTask(subtask_id="st-1", agent="etl", description="Load")],
        output_format=OutputFormat.JSON,
        complexity=TaskComplexity.LOW,
    )


@pytest.fixture
def etl_result():
    return ETLResult(
        task_id="json-test-001",
        source_ids=["src-001"],
        row_count=500,
        column_count=3,
        schema=[
            ColumnSchema(name="x", dtype="Float64", nullable=False, null_rate=0.0),
            ColumnSchema(name="y", dtype="Int64", nullable=False, null_rate=0.0),
            ColumnSchema(name="label", dtype="Utf8", nullable=True, null_rate=0.05),
        ],
        validation_issues=[
            ValidationIssue(column="label", severity="warning",
                            message="5% null values", affected_rows=25),
        ],
        elapsed_seconds=0.75,
    )


@pytest.fixture
def analysis_result():
    return AnalysisResult(
        task_id="json-test-001",
        summary_stats=[
            SummaryStats(column="x", dtype="Float64", count=500, null_count=0,
                         mean=10.5, std=2.1, min=1.0, max=20.0,
                         p25=8.5, p50=10.5, p75=12.5),
        ],
        anomalies=[
            AnomalyRecord(row_index=99, column="x", value=99.9,
                          anomaly_score=-0.55, method="ensemble"),
        ],
        anomaly_rate=0.002,
        key_findings=["One outlier detected at row 99."],
    )


@pytest.fixture
def cost_summary():
    return CostSummary.from_entries(
        task_id="json-test-001",
        entries=[
            CostEntry(task_id="json-test-001", agent_id="etl",
                      model="claude-haiku-4-5-20251001",
                      input_tokens=400, output_tokens=200, cost_usd=0.0004, latency_ms=300.0),
        ],
    )


# ─── emit_json ────────────────────────────────────────────────────────────────


class TestEmitJSON:
    def test_returns_string(self, task_plan, etl_result, analysis_result):
        result = emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY)
        assert isinstance(result, str)

    def test_valid_json(self, task_plan, etl_result, analysis_result):
        result = emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_meta_section_present(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert "meta" in parsed
        assert parsed["meta"]["task_id"] == "json-test-001"

    def test_executive_summary_present(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert parsed["executive_summary"] == EXEC_SUMMARY

    def test_etl_section_present(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert "etl" in parsed
        assert parsed["etl"]["row_count"] == 500
        assert parsed["etl"]["column_count"] == 3

    def test_etl_schema_serialised(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        schema = parsed["etl"]["schema"]
        assert len(schema) == 3
        col_names = [s["name"] for s in schema]
        assert "x" in col_names

    def test_etl_validation_issues_serialised(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        issues = parsed["etl"]["validation_issues"]
        assert len(issues) == 1
        assert issues[0]["severity"] == "warning"

    def test_analysis_section_present(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert "analysis" in parsed
        assert parsed["analysis"]["anomaly_count"] == 1
        assert parsed["analysis"]["anomaly_rate"] == pytest.approx(0.002)

    def test_anomalies_serialised(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        anomalies = parsed["analysis"]["anomalies"]
        assert len(anomalies) == 1
        assert anomalies[0]["row_index"] == 99
        assert anomalies[0]["method"] == "ensemble"

    def test_summary_stats_serialised(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        stats = parsed["analysis"]["summary_stats"]
        assert len(stats) == 1
        assert stats[0]["mean"] == pytest.approx(10.5)

    def test_key_findings_serialised(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert parsed["analysis"]["key_findings"] == ["One outlier detected at row 99."]

    def test_cost_section_present_when_provided(self, task_plan, etl_result,
                                                 analysis_result, cost_summary):
        parsed = json.loads(emit_json(
            task_plan, etl_result, analysis_result, EXEC_SUMMARY,
            cost_summary=cost_summary,
        ))
        assert "cost" in parsed
        assert parsed["cost"]["total_cost_usd"] == pytest.approx(0.0004)

    def test_cost_absent_when_not_provided(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert "cost" not in parsed

    def test_errors_section_when_provided(self, task_plan, etl_result, analysis_result):
        errors = [AgentErrorRecord(
            agent_id="etl", attempt=2, error_type="TimeoutError",
            message="Timed out", recovery_tier=RecoveryTier.RETRY,
        )]
        parsed = json.loads(emit_json(
            task_plan, etl_result, analysis_result, EXEC_SUMMARY, errors=errors
        ))
        assert "errors" in parsed
        assert parsed["errors"][0]["agent_id"] == "etl"

    def test_no_errors_key_when_none(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert "errors" not in parsed

    def test_generated_at_is_iso_format(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        ts = parsed["meta"]["generated_at"]
        assert "T" in ts  # ISO 8601 format

    def test_indented_output(self, task_plan, etl_result, analysis_result):
        result = emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY, indent=2)
        assert "\n" in result  # multi-line = indented

    def test_task_type_in_meta(self, task_plan, etl_result, analysis_result):
        parsed = json.loads(emit_json(task_plan, etl_result, analysis_result, EXEC_SUMMARY))
        assert parsed["meta"]["task_type"] == "full_pipeline"


# ─── PDF renderer availability check ─────────────────────────────────────────


class TestPDFRendererAvailability:
    def test_is_available_returns_bool(self):
        result = pdf_available()
        assert isinstance(result, bool)

    def test_render_pdf_raises_import_error_when_unavailable(self, tmp_path):
        """If WeasyPrint is not installed, render_pdf raises ImportError."""
        from pipeline.agents.report.tools import pdf_renderer
        original = pdf_renderer._WEASYPRINT_AVAILABLE
        try:
            pdf_renderer._WEASYPRINT_AVAILABLE = False
            from pipeline.agents.report.tools.pdf_renderer import render_pdf
            with pytest.raises(ImportError, match="WeasyPrint"):
                render_pdf("# Test", str(tmp_path / "test.pdf"))
        finally:
            pdf_renderer._WEASYPRINT_AVAILABLE = original
