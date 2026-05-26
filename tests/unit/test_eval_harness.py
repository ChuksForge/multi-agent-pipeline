"""
tests/unit/test_eval_harness.py
─────────────────────────────────
Unit tests for the eval harness: fixture loading, expected output parsing,
and result writing. Pipeline runs are NOT invoked — harness logic only.
"""

from __future__ import annotations

import json
import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from evals.harness import load_fixtures, parse_expected
from evals.metrics import ExpectedOutput


# ─── load_fixtures ────────────────────────────────────────────────────────────


class TestLoadFixtures:
    def test_returns_list(self, tmp_path):
        result = load_fixtures(tmp_path)
        assert isinstance(result, list)

    def test_empty_dir_returns_empty(self, tmp_path):
        result = load_fixtures(tmp_path)
        assert result == []

    def test_missing_dir_returns_empty(self, tmp_path):
        result = load_fixtures(tmp_path / "nonexistent")
        assert result == []

    def test_loads_valid_fixture_files(self, tmp_path):
        for i in range(3):
            (tmp_path / f"task_00{i+1}.json").write_text(json.dumps({
                "task_id": f"task_00{i+1}",
                "raw_task": f"Task {i+1}",
                "expected": {"status": "complete"},
            }))
        result = load_fixtures(tmp_path)
        assert len(result) == 3

    def test_ignores_non_json_files(self, tmp_path):
        (tmp_path / "task_001.json").write_text(json.dumps({"task_id": "t1", "raw_task": "x"}))
        (tmp_path / "README.md").write_text("# readme")
        (tmp_path / "task_002.txt").write_text("not json")
        result = load_fixtures(tmp_path)
        assert len(result) == 1

    def test_ignores_non_task_json_files(self, tmp_path):
        (tmp_path / "task_001.json").write_text(json.dumps({"task_id": "t1", "raw_task": "x"}))
        (tmp_path / "config.json").write_text(json.dumps({"key": "value"}))
        result = load_fixtures(tmp_path)
        assert len(result) == 1  # only task_*.json files

    def test_sorted_by_name(self, tmp_path):
        for name in ["task_003", "task_001", "task_002"]:
            (tmp_path / f"{name}.json").write_text(
                json.dumps({"task_id": name, "raw_task": name})
            )
        result = load_fixtures(tmp_path)
        ids = [f["task_id"] for f in result]
        assert ids == sorted(ids)

    def test_adds_fixture_path_key(self, tmp_path):
        (tmp_path / "task_001.json").write_text(
            json.dumps({"task_id": "task_001", "raw_task": "test"})
        )
        result = load_fixtures(tmp_path)
        assert "_fixture_path" in result[0]

    def test_skips_malformed_json_gracefully(self, tmp_path):
        (tmp_path / "task_001.json").write_text('{"valid": true, "task_id": "t1", "raw_task": "x"}')
        (tmp_path / "task_002.json").write_text("{ broken json }")
        result = load_fixtures(tmp_path)
        # Only the valid fixture is loaded
        assert len(result) == 1
        assert result[0]["task_id"] == "t1"


# ─── parse_expected ───────────────────────────────────────────────────────────


class TestParseExpected:
    def test_returns_expected_output(self):
        fixture = {"expected": {"status": "complete", "min_rows": 100}}
        result = parse_expected(fixture)
        assert isinstance(result, ExpectedOutput)

    def test_status_parsed(self):
        fixture = {"expected": {"status": "partial"}}
        result = parse_expected(fixture)
        assert result.status == "partial"

    def test_min_rows_parsed(self):
        fixture = {"expected": {"min_rows": 500}}
        result = parse_expected(fixture)
        assert result.min_rows == 500

    def test_required_columns_parsed(self):
        fixture = {"expected": {"required_columns": ["date", "revenue"]}}
        result = parse_expected(fixture)
        assert result.required_columns == ["date", "revenue"]

    def test_max_anomaly_rate_parsed(self):
        fixture = {"expected": {"max_anomaly_rate": 0.05}}
        result = parse_expected(fixture)
        assert result.max_anomaly_rate == pytest.approx(0.05)

    def test_min_anomaly_count_parsed(self):
        fixture = {"expected": {"min_anomaly_count": 3}}
        result = parse_expected(fixture)
        assert result.min_anomaly_count == 3

    def test_required_sections_parsed(self):
        fixture = {"expected": {"required_report_sections": ["Executive Summary"]}}
        result = parse_expected(fixture)
        assert "Executive Summary" in result.required_report_sections

    def test_min_word_count_parsed(self):
        fixture = {"expected": {"min_word_count": 200}}
        result = parse_expected(fixture)
        assert result.min_word_count == 200

    def test_allow_partial_parsed(self):
        fixture = {"expected": {"allow_partial": True}}
        result = parse_expected(fixture)
        assert result.allow_partial is True

    def test_missing_expected_block_uses_defaults(self):
        fixture = {"task_id": "t1", "raw_task": "test"}
        result = parse_expected(fixture)
        assert result.status == "complete"
        assert result.min_rows == 0
        assert result.max_anomaly_rate == pytest.approx(1.0)

    def test_empty_expected_block_uses_defaults(self):
        fixture = {"expected": {}}
        result = parse_expected(fixture)
        assert isinstance(result, ExpectedOutput)
        assert result.required_columns == []
