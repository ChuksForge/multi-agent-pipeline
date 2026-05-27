"""
api/routes/pipeline.py
───────────────────────
Pipeline API routes.

POST /run              — Submit a task. Runs synchronously (returns when done).
GET  /status/{task_id} — Lightweight status check (from in-memory store).
GET  /result/{task_id} — Full result with report content.
GET  /runs             — List recent runs (last 50).
DELETE /runs/{task_id} — Remove a run from the store.

Design note on synchronous execution:
  The pipeline runs synchronously within the request for simplicity.
  For production, replace with a background task queue (Celery, ARQ, etc.)
  and return 202 Accepted immediately. The status/result endpoints already
  support async polling — the store interface is the same either way.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from pipeline.api.dependencies import RequestIdDep, get_graph
from pipeline.api.models import (
    ErrorResponse,
    ResultResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
)
from pipeline.core.schemas import CostSummary, PipelineStatus
from pipeline.core.state import PipelineState, initial_state
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["pipeline"])

# ── In-memory run store ───────────────────────────────────────────────────────
# For production: replace with Redis or a database.
# Keyed by task_id → final PipelineState dict.

_run_store: dict[str, dict[str, Any]] = {}
_MAX_STORED_RUNS = 100


def _store_run(task_id: str, state: PipelineState, elapsed: float) -> None:
    """Store a completed run. Evicts oldest entry when store is full."""
    _run_store[task_id] = {
        "state": state,
        "elapsed_seconds": elapsed,
        "stored_at": datetime.now(timezone.utc),
    }
    # Evict oldest when over limit
    if len(_run_store) > _MAX_STORED_RUNS:
        oldest = next(iter(_run_store))
        del _run_store[oldest]


def _get_run(task_id: str) -> dict[str, Any]:
    """Retrieve a run or raise 404."""
    entry = _run_store.get(task_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found. It may still be running or never existed.",
        )
    return entry


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post(
    "/run",
    response_model=RunResponse,
    status_code=status.HTTP_200_OK,
    summary="Run a pipeline task",
    description=(
        "Submit a natural language data task. The pipeline runs synchronously "
        "and returns when complete. Use GET /result/{task_id} to retrieve the report."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid task description"},
        500: {"model": ErrorResponse, "description": "Pipeline execution failed"},
    },
)
async def run_pipeline(
    request: RunRequest,
    request_id: RequestIdDep,
    graph=Depends(get_graph),
) -> RunResponse:
    """
    Execute the full pipeline: Planner → ETL → Analysis → Report.

    The task string is passed directly to the Planner agent which decomposes
    it into a structured execution plan. Data sources mentioned in the task
    are loaded automatically by the ETL agent.
    """
    task_id = request.task_id or str(uuid.uuid4())
    raw_task = request.task

    # Append output format hint if not markdown
    if request.output_format != "markdown":
        raw_task += f" Output format: {request.output_format}."

    logger.info(
        "api_run_start",
        task_id=task_id,
        request_id=request_id,
        output_format=request.output_format,
    )

    import time
    start = time.monotonic()

    try:
        start_state = initial_state(task_id=task_id, raw_task=raw_task)
        final_state: PipelineState = await graph.ainvoke(start_state)
        elapsed = time.monotonic() - start

        _store_run(task_id, final_state, elapsed)

        pipeline_status = final_state.get("status", PipelineStatus.FAILED)
        logger.info(
            "api_run_complete",
            task_id=task_id,
            status=pipeline_status.value,
            elapsed_s=round(elapsed, 2),
        )

        return RunResponse(
            task_id=task_id,
            status=pipeline_status.value,
            message=f"Pipeline completed in {elapsed:.1f}s",
        )

    except Exception as e:
        elapsed = time.monotonic() - start
        logger.exception("api_run_failed", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline execution failed: {e}",
        )


@router.get(
    "/status/{task_id}",
    response_model=StatusResponse,
    summary="Get pipeline status",
    responses={404: {"model": ErrorResponse}},
)
async def get_status(task_id: str) -> StatusResponse:
    """Lightweight status check — returns state summary without report content."""
    entry = _get_run(task_id)
    state: PipelineState = entry["state"]

    pipeline_status = state.get("status", PipelineStatus.FAILED)
    cost_entries = state.get("cost_log", [])
    total_cost = sum(e.cost_usd for e in cost_entries)
    errors = state.get("errors", [])

    return StatusResponse(
        task_id=task_id,
        status=pipeline_status.value,
        current_agent=state.get("current_agent"),
        has_etl_result=state.get("etl_result") is not None,
        has_analysis_result=state.get("analysis_result") is not None,
        has_report_result=state.get("report_result") is not None,
        error_count=len(errors),
        total_cost_usd=round(total_cost, 6),
        elapsed_seconds=entry.get("elapsed_seconds"),
        created_at=entry.get("stored_at"),
    )


@router.get(
    "/result/{task_id}",
    response_model=ResultResponse,
    summary="Get full pipeline result",
    description="Returns the complete pipeline result including the report content.",
    responses={404: {"model": ErrorResponse}},
)
async def get_result(task_id: str) -> ResultResponse:
    """Full result — includes report content, anomalies, cost breakdown."""
    entry = _get_run(task_id)
    state: PipelineState = entry["state"]

    pipeline_status = state.get("status", PipelineStatus.FAILED)
    etl = state.get("etl_result")
    analysis = state.get("analysis_result")
    report = state.get("report_result")
    cost_entries = state.get("cost_log", [])
    errors = state.get("errors", [])

    cost_sum = CostSummary.from_entries(task_id=task_id, entries=cost_entries)

    return ResultResponse(
        task_id=task_id,
        status=pipeline_status.value,
        # ETL
        row_count=etl.row_count if etl else 0,
        column_count=etl.column_count if etl else 0,
        validation_error_count=sum(
            1 for i in (etl.validation_issues if etl else []) if i.severity == "error"
        ),
        validation_warning_count=sum(
            1 for i in (etl.validation_issues if etl else []) if i.severity == "warning"
        ),
        # Analysis
        anomaly_count=analysis.anomaly_count if analysis else 0,
        anomaly_rate=analysis.anomaly_rate if analysis else 0.0,
        key_findings=analysis.key_findings if analysis else [],
        chart_count=len(analysis.charts) if analysis else 0,
        # Report
        output_format=report.output_format.value if report else "markdown",
        report_content=report.full_content if report else "",
        word_count=report.word_count if report else 0,
        # Cost
        total_cost_usd=cost_sum.total_cost_usd,
        total_input_tokens=cost_sum.total_input_tokens,
        total_output_tokens=cost_sum.total_output_tokens,
        cost_per_agent=cost_sum.per_agent,
        # Errors
        errors=[
            {
                "agent_id": e.agent_id,
                "error_type": e.error_type,
                "message": e.message,
                "recovery_tier": e.recovery_tier.value,
            }
            for e in errors
        ],
    )


@router.get(
    "/runs",
    summary="List recent runs",
    description=f"Returns the last {_MAX_STORED_RUNS} pipeline runs (in-memory store).",
)
async def list_runs() -> list[dict[str, Any]]:
    """List all stored pipeline runs, most recent first."""
    runs = []
    for task_id, entry in reversed(list(_run_store.items())):
        state: PipelineState = entry["state"]
        pipeline_status = state.get("status", PipelineStatus.FAILED)
        cost_entries = state.get("cost_log", [])
        total_cost = sum(e.cost_usd for e in cost_entries)
        runs.append({
            "task_id": task_id,
            "status": pipeline_status.value,
            "elapsed_seconds": entry.get("elapsed_seconds"),
            "stored_at": entry.get("stored_at"),
            "total_cost_usd": round(total_cost, 6),
            "has_report": state.get("report_result") is not None,
        })
    return runs


@router.delete(
    "/runs/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a run",
    responses={404: {"model": ErrorResponse}},
)
async def delete_run(task_id: str) -> None:
    """Remove a run from the in-memory store."""
    _get_run(task_id)  # raises 404 if not found
    del _run_store[task_id]
    logger.info("api_run_deleted", task_id=task_id)
