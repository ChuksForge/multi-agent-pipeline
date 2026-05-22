"""
tests/unit/test_planner.py
───────────────────────────
Unit tests for the Task Planner.
LLM call is mocked — tests validate JSON parsing, Pydantic validation,
error handling, fallback defaults, and retry logic.
"""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from pipeline.core.exceptions import InvalidTaskError, TaskPlanningError
from pipeline.core.schemas import (
    OutputFormat,
    TaskComplexity,
    TaskType,
)
from pipeline.planner.planner import (
    _build_task_plan,
    _default_subtasks,
    _parse_json,
    plan_task,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


VALID_PLAN_JSON = {
    "raw_task": "Analyse sales data for anomalies",
    "data_sources": [{"uri": "data/sales.csv", "table_name": "sales"}],
    "task_type": "full_pipeline",
    "output_format": "markdown",
    "complexity": "medium",
    "subtasks": [
        {"subtask_id": "st-001", "agent": "etl",
         "description": "Load CSV", "depends_on": [], "required": True,
         "estimated_tokens": 500},
        {"subtask_id": "st-002", "agent": "analysis",
         "description": "Detect anomalies", "depends_on": ["st-001"],
         "required": True, "estimated_tokens": 800},
        {"subtask_id": "st-003", "agent": "report",
         "description": "Generate report", "depends_on": ["st-002"],
         "required": True, "estimated_tokens": 400},
    ],
}


def _mock_llm_response(plan_dict: dict) -> MagicMock:
    mock = MagicMock()
    mock.content = json.dumps(plan_dict)
    return mock


def _mock_llm_response_fenced(plan_dict: dict) -> MagicMock:
    mock = MagicMock()
    mock.content = f"```json\n{json.dumps(plan_dict)}\n```"
    return mock


# ─── _parse_json ──────────────────────────────────────────────────────────────


class TestParseJSON:
    def test_parses_clean_json(self):
        result = _parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_strips_json_fences(self):
        result = _parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_strips_plain_fences(self):
        result = _parse_json('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(TaskPlanningError, match="invalid JSON"):
            _parse_json("not json at all")

    def test_non_dict_raises(self):
        with pytest.raises(TaskPlanningError, match="Expected JSON object"):
            _parse_json("[1, 2, 3]")

    def test_whitespace_trimmed(self):
        result = _parse_json('  \n  {"a": 1}  \n  ')
        assert result == {"a": 1}


# ─── _build_task_plan ─────────────────────────────────────────────────────────


class TestBuildTaskPlan:
    def test_valid_plan_builds(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test task", "task-001")
        assert plan.task_id == "task-001"
        assert plan.task_type == TaskType.FULL_PIPELINE
        assert len(plan.subtasks) == 3

    def test_raw_task_preserved(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "original task", "t-001")
        assert plan.raw_task == "original task"

    def test_data_sources_parsed(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test", "t-001")
        assert len(plan.data_sources) == 1
        assert plan.data_sources[0].uri == "data/sales.csv"
        assert plan.data_sources[0].table_name == "sales"

    def test_string_data_source_accepted(self):
        data = dict(VALID_PLAN_JSON)
        data["data_sources"] = ["data/file.csv"]
        plan = _build_task_plan(data, "test", "t-001")
        assert plan.data_sources[0].uri == "data/file.csv"

    def test_output_format_parsed(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test", "t-001")
        assert plan.output_format == OutputFormat.MARKDOWN

    def test_pdf_output_format(self):
        data = {**VALID_PLAN_JSON, "output_format": "pdf"}
        plan = _build_task_plan(data, "test", "t-001")
        assert plan.output_format == OutputFormat.PDF

    def test_complexity_parsed(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test", "t-001")
        assert plan.complexity == TaskComplexity.MEDIUM

    def test_invalid_task_type_defaults_to_full_pipeline(self):
        data = {**VALID_PLAN_JSON, "task_type": "invalid_type"}
        plan = _build_task_plan(data, "test", "t-001")
        assert plan.task_type == TaskType.FULL_PIPELINE

    def test_invalid_output_format_defaults_to_markdown(self):
        data = {**VALID_PLAN_JSON, "output_format": "docx"}
        plan = _build_task_plan(data, "test", "t-001")
        assert plan.output_format == OutputFormat.MARKDOWN

    def test_missing_subtasks_uses_defaults(self):
        data = {**VALID_PLAN_JSON, "subtasks": []}
        plan = _build_task_plan(data, "test", "t-001")
        assert len(plan.subtasks) >= 1

    def test_dependencies_preserved(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test", "t-001")
        analysis_task = next(s for s in plan.subtasks if s.agent == "analysis")
        assert "st-001" in analysis_task.depends_on

    def test_estimated_tokens_set(self):
        plan = _build_task_plan(VALID_PLAN_JSON, "test", "t-001")
        etl_task = next(s for s in plan.subtasks if s.agent == "etl")
        assert etl_task.estimated_tokens == 500

    def test_empty_data_sources_allowed(self):
        data = {**VALID_PLAN_JSON, "data_sources": []}
        plan = _build_task_plan(data, "test", "t-001")
        assert plan.data_sources == []


# ─── _default_subtasks ────────────────────────────────────────────────────────


class TestDefaultSubtasks:
    def test_full_pipeline_has_three_subtasks(self):
        result = _default_subtasks(TaskType.FULL_PIPELINE)
        assert len(result) == 3
        agents = [s["agent"] for s in result]
        assert agents == ["etl", "analysis", "report"]

    def test_etl_only_has_one_subtask(self):
        result = _default_subtasks(TaskType.ETL_ONLY)
        assert len(result) == 1
        assert result[0]["agent"] == "etl"

    def test_analysis_only(self):
        result = _default_subtasks(TaskType.ANALYSIS_ONLY)
        assert result[0]["agent"] == "analysis"

    def test_report_only(self):
        result = _default_subtasks(TaskType.REPORT_ONLY)
        assert result[0]["agent"] == "report"

    def test_full_pipeline_has_correct_dependencies(self):
        result = _default_subtasks(TaskType.FULL_PIPELINE)
        analysis = next(s for s in result if s["agent"] == "analysis")
        report = next(s for s in result if s["agent"] == "report")
        assert "st-001" in analysis["depends_on"]
        assert "st-002" in report["depends_on"]


# ─── plan_task (LLM mocked) ───────────────────────────────────────────────────


class TestPlanTask:
    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_returns_task_plan(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        plan = plan_task("Analyse sales data for anomalies")
        from pipeline.core.schemas import TaskPlan
        assert isinstance(plan, TaskPlan)

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_task_id_generated_if_not_provided(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        plan = plan_task("Analyse sales data")
        assert len(plan.task_id) > 0

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_task_id_override_respected(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        plan = plan_task("Analyse sales data", task_id="my-task-999")
        assert plan.task_id == "my-task-999"

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_fenced_json_accepted(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response_fenced(VALID_PLAN_JSON)
        plan = plan_task("Analyse data")
        assert len(plan.subtasks) == 3

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_subtasks_populated(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        plan = plan_task("Analyse sales data")
        assert len(plan.subtasks) == 3

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_data_sources_populated(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        plan = plan_task("Analyse sales data")
        assert len(plan.data_sources) == 1

    def test_empty_task_raises_invalid_task_error(self):
        with pytest.raises(InvalidTaskError):
            plan_task("")

    def test_too_short_task_raises(self):
        with pytest.raises(InvalidTaskError):
            plan_task("hi")

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_invalid_json_response_raises_planning_error(self, mock_cls):
        mock_response = MagicMock()
        mock_response.content = "Sorry, I cannot help with that."
        mock_cls.return_value.invoke.return_value = mock_response
        with pytest.raises(TaskPlanningError):
            plan_task("Analyse sales data")

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_llm_exception_raises_planning_error(self, mock_cls):
        mock_cls.return_value.invoke.side_effect = RuntimeError("API down")
        with pytest.raises(TaskPlanningError):
            plan_task("Analyse sales data")

    @patch("pipeline.planner.planner.ChatAnthropic")
    def test_raw_task_preserved_in_plan(self, mock_cls):
        mock_cls.return_value.invoke.return_value = _mock_llm_response(VALID_PLAN_JSON)
        raw = "Analyse monthly sales data for revenue anomalies"
        plan = plan_task(raw)
        assert plan.raw_task == raw
