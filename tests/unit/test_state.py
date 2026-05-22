"""
tests/unit/test_state.py
────────────────────────
Tests for PipelineState: initial construction, helper functions,
and the append-only reducer behaviour.
"""

from __future__ import annotations

import pytest

from pipeline.core.schemas import (
    AgentErrorRecord,
    CostEntry,
    PipelineStatus,
    RecoveryTier,
)
from pipeline.core.state import (
    PipelineState,
    agent_retry_count,
    has_analysis_data,
    has_etl_data,
    increment_retry,
    initial_state,
    is_failed,
    mark_agent_skipped,
)


class TestInitialState:
    def test_all_results_none(self, task_id):
        state = initial_state(task_id=task_id, raw_task="test")
        assert state["etl_result"] is None
        assert state["analysis_result"] is None
        assert state["report_result"] is None
        assert state["task_plan"] is None

    def test_lists_are_empty(self, task_id):
        state = initial_state(task_id=task_id, raw_task="test")
        assert state["errors"] == []
        assert state["cost_log"] == []
        assert state["skip_agents"] == []

    def test_status_is_pending(self, task_id):
        state = initial_state(task_id=task_id, raw_task="test")
        assert state["status"] == PipelineStatus.PENDING

    def test_retry_counts_empty(self, task_id):
        state = initial_state(task_id=task_id, raw_task="test")
        assert state["retry_counts"] == {}

    def test_task_id_preserved(self, task_id):
        state = initial_state(task_id=task_id, raw_task="test")
        assert state["task_id"] == task_id

    def test_raw_task_preserved(self, task_id):
        raw = "Analyse sales data for anomalies"
        state = initial_state(task_id=task_id, raw_task=raw)
        assert state["raw_task"] == raw


class TestHelperFunctions:
    def test_has_etl_data_false_when_none(self, empty_state):
        assert has_etl_data(empty_state) is False

    def test_has_etl_data_true_after_etl(self, state_after_etl):
        assert has_etl_data(state_after_etl) is True

    def test_has_etl_data_false_on_zero_rows(self, empty_state, task_id):
        from pipeline.core.schemas import ETLResult
        # TypedDict is just a dict at runtime — mutate directly, no constructor
        state: PipelineState = dict(empty_state)  # type: ignore[assignment]
        state["etl_result"] = ETLResult(
            task_id=task_id, source_ids=[], row_count=0, column_count=0, schema=[]
        )
        assert has_etl_data(state) is False

    def test_has_analysis_data_false_when_none(self, empty_state):
        assert has_analysis_data(empty_state) is False

    def test_has_analysis_data_true(self, empty_state, clean_analysis_result):
        state: PipelineState = dict(empty_state)  # type: ignore[assignment]
        state["analysis_result"] = clean_analysis_result
        assert has_analysis_data(state) is True

    def test_is_failed_false_initially(self, empty_state):
        assert is_failed(empty_state) is False

    def test_is_failed_true(self, empty_state):
        state: PipelineState = dict(empty_state)  # type: ignore[assignment]
        state["status"] = PipelineStatus.FAILED
        assert is_failed(state) is True

    def test_agent_retry_count_zero_for_new_agent(self, empty_state):
        assert agent_retry_count(empty_state, "etl") == 0

    def test_increment_retry_first_time(self, empty_state):
        update = increment_retry(empty_state, "etl")
        assert update["retry_counts"]["etl"] == 1

    def test_increment_retry_accumulates(self, empty_state):
        update1 = increment_retry(empty_state, "etl")
        # TypedDict is a plain dict at runtime — mutate directly
        state: PipelineState = dict(empty_state)  # type: ignore[assignment]
        state["retry_counts"] = update1["retry_counts"]
        update2 = increment_retry(state, "etl")
        assert update2["retry_counts"]["etl"] == 2

    def test_increment_retry_does_not_affect_other_agents(self, empty_state):
        update = increment_retry(empty_state, "etl")
        assert "analysis" not in update["retry_counts"]

    def test_mark_agent_skipped(self, empty_state):
        update = mark_agent_skipped(empty_state, "analysis")
        assert "analysis" in update["skip_agents"]

    def test_mark_agent_skipped_idempotent(self, empty_state):
        update1 = mark_agent_skipped(empty_state, "analysis")
        state: PipelineState = dict(empty_state)  # type: ignore[assignment]
        state["skip_agents"] = update1["skip_agents"]
        update2 = mark_agent_skipped(state, "analysis")
        assert update2["skip_agents"].count("analysis") == 1
