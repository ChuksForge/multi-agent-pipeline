"""
tests/unit/test_token_tracker.py
─────────────────────────────────
Tests for the TokenTracker LangChain callback handler.
Uses mock LLMResult objects to simulate real LLM responses.
"""

from __future__ import annotations

import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from pipeline.middleware.token_tracker import TokenTracker, make_tracker


def _make_llm_result(
    model: str = "claude-haiku-4-5-20251001",
    input_tokens: int = 500,
    output_tokens: int = 300,
) -> MagicMock:
    """Build a mock LLMResult with the usage structure Anthropic provides."""
    result = MagicMock()
    result.llm_output = {
        "model": model,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }
    # Fallback gen_info path
    gen = MagicMock()
    gen.generation_info = {}
    result.generations = [[gen]]
    return result


def _make_openai_result(
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 400,
    completion_tokens: int = 200,
) -> MagicMock:
    """Build a mock LLMResult with the OpenAI token field names."""
    result = MagicMock()
    result.llm_output = {
        "model": model,
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }
    gen = MagicMock()
    gen.generation_info = {}
    result.generations = [[gen]]
    return result


class TestTokenTrackerBasic:
    def test_initial_state_empty(self, token_tracker):
        assert token_tracker.entries == []
        assert token_tracker.total_cost_usd == pytest.approx(0.0)
        assert token_tracker.total_input_tokens == 0
        assert token_tracker.total_output_tokens == 0

    def test_make_tracker_factory(self, task_id):
        tracker = make_tracker(task_id=task_id, agent_id="analysis")
        assert tracker.task_id == task_id
        assert tracker.agent_id == "analysis"


class TestOnLLMEnd:
    def test_records_entry_on_llm_end(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["prompt"], run_id=run_id)
        result = _make_llm_result(input_tokens=500, output_tokens=300)
        token_tracker.on_llm_end(result, run_id=run_id)

        assert len(token_tracker.entries) == 1
        entry = token_tracker.entries[0]
        assert entry.input_tokens == 500
        assert entry.output_tokens == 300
        assert entry.agent_id == token_tracker.agent_id
        assert entry.task_id == token_tracker.task_id

    def test_cost_is_positive(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["prompt"], run_id=run_id)
        token_tracker.on_llm_end(_make_llm_result(input_tokens=1000, output_tokens=500), run_id=run_id)
        assert token_tracker.entries[0].cost_usd > 0.0

    def test_latency_is_recorded(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["prompt"], run_id=run_id)
        time.sleep(0.01)  # 10ms sleep
        token_tracker.on_llm_end(_make_llm_result(), run_id=run_id)
        assert token_tracker.entries[0].latency_ms >= 10.0

    def test_multiple_calls_accumulate(self, token_tracker):
        for i in range(3):
            run_id = uuid.uuid4()
            token_tracker.on_llm_start({}, ["p"], run_id=run_id)
            token_tracker.on_llm_end(_make_llm_result(input_tokens=100, output_tokens=50), run_id=run_id)
        assert len(token_tracker.entries) == 3

    def test_openai_token_field_names(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["prompt"], run_id=run_id)
        token_tracker.on_llm_end(_make_openai_result(prompt_tokens=400, completion_tokens=200), run_id=run_id)
        entry = token_tracker.entries[0]
        assert entry.input_tokens == 400
        assert entry.output_tokens == 200

    def test_model_name_captured(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["p"], run_id=run_id)
        token_tracker.on_llm_end(_make_llm_result(model="claude-haiku-4-5-20251001"), run_id=run_id)
        assert token_tracker.entries[0].model == "claude-haiku-4-5-20251001"


class TestOnLLMError:
    def test_error_clears_timing_entry(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["p"], run_id=run_id)
        token_tracker.on_llm_error(RuntimeError("timeout"), run_id=run_id)
        # Timing entry should be cleaned up — no stale key
        assert str(run_id) not in token_tracker._call_start_times

    def test_error_does_not_add_cost_entry(self, token_tracker):
        run_id = uuid.uuid4()
        token_tracker.on_llm_start({}, ["p"], run_id=run_id)
        token_tracker.on_llm_error(RuntimeError("timeout"), run_id=run_id)
        assert len(token_tracker.entries) == 0


class TestAggregation:
    def test_total_cost_sums_entries(self, populated_tracker):
        expected = sum(e.cost_usd for e in populated_tracker.entries)
        assert populated_tracker.total_cost_usd == pytest.approx(expected)

    def test_total_input_tokens(self, populated_tracker):
        expected = sum(e.input_tokens for e in populated_tracker.entries)
        assert populated_tracker.total_input_tokens == expected

    def test_total_output_tokens(self, populated_tracker):
        expected = sum(e.output_tokens for e in populated_tracker.entries)
        assert populated_tracker.total_output_tokens == expected

    def test_total_latency(self, populated_tracker):
        expected = sum(e.latency_ms for e in populated_tracker.entries)
        assert populated_tracker.total_latency_ms == pytest.approx(expected)

    def test_summary_dict_keys(self, populated_tracker):
        summary = populated_tracker.summary()
        required_keys = {"agent_id", "calls", "total_input_tokens", "total_output_tokens", "total_cost_usd", "total_latency_ms"}
        assert required_keys.issubset(set(summary.keys()))

    def test_summary_call_count(self, populated_tracker):
        assert populated_tracker.summary()["calls"] == 2


class TestNoStartNoLatency:
    def test_llm_end_without_start_records_zero_latency(self, token_tracker):
        """Should not crash if on_llm_start was somehow missed."""
        run_id = uuid.uuid4()
        token_tracker.on_llm_end(_make_llm_result(), run_id=run_id)
        assert token_tracker.entries[0].latency_ms == pytest.approx(0.0)
