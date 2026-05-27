"""
api/models.py
──────────────
FastAPI request and response Pydantic models.

Separate from core/schemas.py — these are the HTTP contract, not the
internal pipeline contract. They're deliberately simpler and more
human-friendly than the internal typed state objects.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────


class RunRequest(BaseModel):
    """POST /run — start a pipeline task."""

    task: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Natural language description of the data task.",
        examples=[
            "Load data/samples/sales_monthly.csv, detect revenue anomalies, "
            "and generate a markdown report."
        ],
    )
    output_format: str = Field(
        default="markdown",
        pattern="^(markdown|json|pdf)$",
        description="Report output format.",
    )
    task_id: str | None = Field(
        default=None,
        description="Optional task ID override. Auto-generated if not provided.",
    )


# ── Response models ───────────────────────────────────────────────────────────


class RunResponse(BaseModel):
    """POST /run response — task accepted."""

    task_id: str
    status: str
    message: str = "Pipeline task started"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class StatusResponse(BaseModel):
    """GET /status/{task_id} — current pipeline state."""

    task_id: str
    status: str
    current_agent: str | None = None
    has_etl_result: bool = False
    has_analysis_result: bool = False
    has_report_result: bool = False
    error_count: int = 0
    total_cost_usd: float = 0.0
    elapsed_seconds: float | None = None
    created_at: datetime | None = None


class ResultResponse(BaseModel):
    """GET /result/{task_id} — full pipeline result."""

    task_id: str
    status: str

    # ETL summary
    row_count: int = 0
    column_count: int = 0
    validation_error_count: int = 0
    validation_warning_count: int = 0

    # Analysis summary
    anomaly_count: int = 0
    anomaly_rate: float = 0.0
    key_findings: list[str] = Field(default_factory=list)
    chart_count: int = 0

    # Report
    output_format: str = "markdown"
    report_content: str = ""
    word_count: int = 0

    # Cost
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost_per_agent: dict[str, float] = Field(default_factory=dict)

    # Errors
    errors: list[dict[str, Any]] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health"""

    status: str = "ok"
    version: str = "0.1.0"
    graph_ready: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    error: str
    detail: str | None = None
    task_id: str | None = None
    request_id: str | None = None
