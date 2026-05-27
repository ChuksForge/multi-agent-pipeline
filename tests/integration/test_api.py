"""
tests/integration/test_api.py
───────────────────────────────
Integration tests for the FastAPI application.

Uses FastAPI's TestClient (synchronous httpx-based client).
The pipeline graph is replaced with a mock that returns a fixed final state,
so no real LLM calls or file I/O happens.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

from fastapi.testclient import TestClient

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
)
from pipeline.core.state import PipelineState, initial_state


# ─── Mock pipeline state ──────────────────────────────────────────────────────


def _make_complete_state(task_id: str) -> PipelineState:
    """Build a realistic final PipelineState for mock graph returns."""
    state: PipelineState = initial_state(task_id=task_id, raw_task="test task")  # type: ignore[assignment]
    state["status"] = PipelineStatus.COMPLETE
    state["etl_result"] = ETLResult(
        task_id=task_id, source_ids=["src-001"],
        row_count=120, column_count=4,
        schema=[
            ColumnSchema(name="date", dtype="Utf8", nullable=False, null_rate=0.0),
            ColumnSchema(name="revenue", dtype="Float64", nullable=False, null_rate=0.0),
        ],
        elapsed_seconds=1.2,
    )
    state["analysis_result"] = AnalysisResult(
        task_id=task_id,
        summary_stats=[
            SummaryStats(column="revenue", dtype="Float64", count=120, null_count=0,
                         mean=1200.0, std=300.0, min=100.0, max=9999.0,
                         p25=900.0, p50=1200.0, p75=1500.0),
        ],
        anomalies=[
            AnomalyRecord(row_index=30, column="revenue", value=9999.0,
                          anomaly_score=-0.5, method="ensemble"),
        ],
        anomaly_rate=0.008,
        key_findings=["One revenue anomaly at row 30."],
    )
    state["report_result"] = ReportResult(
        task_id=task_id,
        output_format=OutputFormat.MARKDOWN,
        title="Test Report",
        full_content=(
            "# Data Pipeline Report\n\n"
            "## Executive Summary\n\nGood dataset.\n\n"
            "## Anomaly Detection\n\n1 anomaly found.\n"
        ),
        word_count=15,
        elapsed_seconds=0.8,
    )
    state["cost_log"] = [
        CostEntry(
            task_id=task_id, agent_id="analysis",
            model="claude-haiku-4-5-20251001",
            input_tokens=600, output_tokens=400,
            cost_usd=0.00072, latency_ms=550.0,
        ),
    ]
    return state


def _make_mock_graph(task_id_override: str | None = None):
    """Return a mock graph whose ainvoke returns a fixed complete state."""
    mock = MagicMock()

    async def fake_ainvoke(state):
        tid = task_id_override or state.get("task_id", str(uuid.uuid4()))
        return _make_complete_state(tid)

    mock.ainvoke = fake_ainvoke
    return mock


# ─── App fixture ──────────────────────────────────────────────────────────────


@pytest.fixture
def client():
    """TestClient with the graph dependency overridden."""
    from pipeline.api.main import create_app
    from pipeline.api.dependencies import get_graph

    app = create_app()
    mock_graph = _make_mock_graph()
    app.dependency_overrides[get_graph] = lambda: mock_graph

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture
def client_with_run(client):
    """Client that has already submitted one run. Returns (client, task_id)."""
    resp = client.post("/api/v1/run", json={
        "task": "Load sales.csv and detect anomalies"
    })
    assert resp.status_code == 200
    task_id = resp.json()["task_id"]
    return client, task_id


# ─── Health endpoint ──────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_health_has_version(self, client):
        resp = client.get("/health")
        assert "version" in resp.json()

    def test_health_has_timestamp(self, client):
        resp = client.get("/health")
        assert "timestamp" in resp.json()


# ─── Root endpoint ────────────────────────────────────────────────────────────


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_docs_link(self, client):
        resp = client.get("/")
        assert "docs" in resp.json()


# ─── POST /api/v1/run ─────────────────────────────────────────────────────────


class TestRunEndpoint:
    def test_run_returns_200(self, client):
        resp = client.post("/api/v1/run", json={
            "task": "Load data/samples/sales_monthly.csv and analyse it"
        })
        assert resp.status_code == 200

    def test_run_returns_task_id(self, client):
        resp = client.post("/api/v1/run", json={"task": "Analyse sales data"})
        data = resp.json()
        assert "task_id" in data
        assert len(data["task_id"]) > 0

    def test_run_returns_status(self, client):
        resp = client.post("/api/v1/run", json={"task": "Analyse sales data"})
        assert "status" in resp.json()

    def test_run_status_is_complete(self, client):
        resp = client.post("/api/v1/run", json={"task": "Analyse sales data"})
        assert resp.json()["status"] == "complete"

    def test_run_custom_task_id(self, client):
        custom_id = "my-task-abc-123"
        resp = client.post("/api/v1/run", json={
            "task": "Analyse sales data",
            "task_id": custom_id,
        })
        assert resp.json()["task_id"] == custom_id

    def test_run_short_task_returns_422(self, client):
        resp = client.post("/api/v1/run", json={"task": "hi"})
        assert resp.status_code == 422

    def test_run_empty_task_returns_422(self, client):
        resp = client.post("/api/v1/run", json={"task": ""})
        assert resp.status_code == 422

    def test_run_json_output_format(self, client):
        resp = client.post("/api/v1/run", json={
            "task": "Analyse sales data",
            "output_format": "json",
        })
        assert resp.status_code == 200

    def test_run_invalid_output_format_returns_422(self, client):
        resp = client.post("/api/v1/run", json={
            "task": "Analyse sales data",
            "output_format": "docx",
        })
        assert resp.status_code == 422

    def test_run_adds_timing_header(self, client):
        resp = client.post("/api/v1/run", json={"task": "Analyse sales data"})
        assert "X-Process-Time-Ms" in resp.headers


# ─── GET /api/v1/status/{task_id} ────────────────────────────────────────────


class TestStatusEndpoint:
    def test_status_returns_200_for_known_task(self, client_with_run):
        client, task_id = client_with_run
        resp = client.get(f"/api/v1/status/{task_id}")
        assert resp.status_code == 200

    def test_status_returns_task_id(self, client_with_run):
        client, task_id = client_with_run
        resp = client.get(f"/api/v1/status/{task_id}")
        assert resp.json()["task_id"] == task_id

    def test_status_shows_complete(self, client_with_run):
        client, task_id = client_with_run
        resp = client.get(f"/api/v1/status/{task_id}")
        assert resp.json()["status"] == "complete"

    def test_status_has_result_flags(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/status/{task_id}").json()
        assert data["has_etl_result"] is True
        assert data["has_analysis_result"] is True
        assert data["has_report_result"] is True

    def test_status_has_cost(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/status/{task_id}").json()
        assert "total_cost_usd" in data
        assert data["total_cost_usd"] >= 0

    def test_status_404_for_unknown_task(self, client):
        resp = client.get("/api/v1/status/nonexistent-task-id-xyz")
        assert resp.status_code == 404


# ─── GET /api/v1/result/{task_id} ────────────────────────────────────────────


class TestResultEndpoint:
    def test_result_returns_200(self, client_with_run):
        client, task_id = client_with_run
        resp = client.get(f"/api/v1/result/{task_id}")
        assert resp.status_code == 200

    def test_result_has_row_count(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert data["row_count"] == 120

    def test_result_has_anomaly_count(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert data["anomaly_count"] == 1

    def test_result_has_report_content(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert "# Data Pipeline Report" in data["report_content"]

    def test_result_has_key_findings(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert len(data["key_findings"]) >= 1

    def test_result_has_cost_breakdown(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert data["total_cost_usd"] >= 0
        assert isinstance(data["cost_per_agent"], dict)

    def test_result_has_word_count(self, client_with_run):
        client, task_id = client_with_run
        data = client.get(f"/api/v1/result/{task_id}").json()
        assert data["word_count"] > 0

    def test_result_404_for_unknown(self, client):
        resp = client.get("/api/v1/result/no-such-task")
        assert resp.status_code == 404


# ─── GET /api/v1/runs ────────────────────────────────────────────────────────


class TestRunsListEndpoint:
    def test_runs_returns_list(self, client):
        resp = client.get("/api/v1/runs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_runs_includes_submitted_run(self, client_with_run):
        client, task_id = client_with_run
        runs = client.get("/api/v1/runs").json()
        task_ids = [r["task_id"] for r in runs]
        assert task_id in task_ids

    def test_runs_has_status_field(self, client_with_run):
        client, task_id = client_with_run
        runs = client.get("/api/v1/runs").json()
        assert all("status" in r for r in runs)

    def test_runs_has_cost_field(self, client_with_run):
        client, task_id = client_with_run
        runs = client.get("/api/v1/runs").json()
        assert all("total_cost_usd" in r for r in runs)


# ─── DELETE /api/v1/runs/{task_id} ───────────────────────────────────────────


class TestDeleteRunEndpoint:
    def test_delete_returns_204(self, client_with_run):
        client, task_id = client_with_run
        resp = client.delete(f"/api/v1/runs/{task_id}")
        assert resp.status_code == 204

    def test_delete_removes_from_store(self, client_with_run):
        client, task_id = client_with_run
        client.delete(f"/api/v1/runs/{task_id}")
        resp = client.get(f"/api/v1/status/{task_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/v1/runs/no-such-task-xyz")
        assert resp.status_code == 404


# ─── OpenAPI schema ───────────────────────────────────────────────────────────


class TestOpenAPISchema:
    def test_openapi_json_accessible(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_has_run_endpoint(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema.get("paths", {})
        assert "/api/v1/run" in paths

    def test_docs_accessible(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
