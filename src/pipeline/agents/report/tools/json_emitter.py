"""
agents/report/tools/json_emitter.py
─────────────────────────────────────
Structured JSON report emitter.

Used as Tier 2 fallback when Markdown generation fails, and as the
primary output format when output_format == OutputFormat.JSON.

Produces a single JSON object with all pipeline results combined.
The structure mirrors the typed schemas exactly — no lossy transformation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    CostSummary,
    ETLResult,
    TaskPlan,
)
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)


def emit_json(
    plan: TaskPlan,
    etl: ETLResult,
    analysis: AnalysisResult,
    executive_summary: str,
    cost_summary: CostSummary | None = None,
    errors: list[AgentErrorRecord] | None = None,
    indent: int = 2,
) -> str:
    """
    Serialise all pipeline results to a structured JSON string.

    Args:
        plan:              TaskPlan metadata.
        etl:               ETLResult from ETL Agent.
        analysis:          AnalysisResult from Analysis Agent.
        executive_summary: LLM-generated summary string.
        cost_summary:      Optional cost breakdown.
        errors:            Optional error records.
        indent:            JSON indentation (default 2).

    Returns:
        JSON string.
    """
    payload: dict[str, Any] = {
        "meta": {
            "task_id": plan.task_id,
            "raw_task": plan.raw_task,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "output_format": plan.output_format.value,
            "task_type": plan.task_type.value,
        },
        "executive_summary": executive_summary,
        "etl": _serialise_etl(etl),
        "analysis": _serialise_analysis(analysis),
    }

    if cost_summary:
        payload["cost"] = _serialise_cost(cost_summary)

    if errors:
        payload["errors"] = [_serialise_error(e) for e in errors]

    result = json.dumps(payload, indent=indent, default=_json_default)

    logger.debug(
        "json_report_emitted",
        task_id=plan.task_id,
        bytes=len(result),
        anomalies=analysis.anomaly_count,
        charts=len(analysis.charts),
    )
    return result


# ── Section serialisers ───────────────────────────────────────────────────────


def _serialise_etl(etl: ETLResult) -> dict[str, Any]:
    return {
        "source_ids": etl.source_ids,
        "row_count": etl.row_count,
        "column_count": etl.column_count,
        "elapsed_seconds": etl.elapsed_seconds,
        "recovery_tier": etl.recovery_tier.value,
        "warnings": etl.warnings,
        "schema": [
            {
                "name": col.name,
                "dtype": col.dtype,
                "nullable": col.nullable,
                "null_rate": col.null_rate,
                "unique_count": col.unique_count,
                "sample_values": col.sample_values,
            }
            for col in etl.schema
        ],
        "validation_issues": [
            {
                "column": issue.column,
                "severity": issue.severity,
                "message": issue.message,
                "affected_rows": issue.affected_rows,
            }
            for issue in etl.validation_issues
        ],
    }


def _serialise_analysis(analysis: AnalysisResult) -> dict[str, Any]:
    return {
        "anomaly_count": analysis.anomaly_count,
        "anomaly_rate": analysis.anomaly_rate,
        "elapsed_seconds": analysis.elapsed_seconds,
        "recovery_tier": analysis.recovery_tier.value,
        "warnings": analysis.warnings,
        "key_findings": analysis.key_findings,
        "summary_stats": [
            {
                "column": s.column,
                "dtype": s.dtype,
                "count": s.count,
                "null_count": s.null_count,
                "mean": s.mean,
                "std": s.std,
                "min": s.min,
                "max": s.max,
                "p25": s.p25,
                "p50": s.p50,
                "p75": s.p75,
                "top_values": s.top_values,
            }
            for s in analysis.summary_stats
        ],
        "anomalies": [
            {
                "row_index": a.row_index,
                "column": a.column,
                "value": a.value,
                "anomaly_score": a.anomaly_score,
                "method": a.method,
            }
            for a in analysis.anomalies
        ],
        "charts": [
            {
                "chart_id": c.chart_id,
                "title": c.title,
                "description": c.description,
                "spec": c.spec,
            }
            for c in analysis.charts
        ],
    }


def _serialise_cost(cost: CostSummary) -> dict[str, Any]:
    return {
        "total_cost_usd": cost.total_cost_usd,
        "total_input_tokens": cost.total_input_tokens,
        "total_output_tokens": cost.total_output_tokens,
        "total_latency_ms": cost.total_latency_ms,
        "per_agent": cost.per_agent,
        "per_model": cost.per_model,
    }


def _serialise_error(error: AgentErrorRecord) -> dict[str, Any]:
    return {
        "agent_id": error.agent_id,
        "attempt": error.attempt,
        "error_type": error.error_type,
        "message": error.message,
        "recovery_tier": error.recovery_tier.value,
        "timestamp": error.timestamp.isoformat(),
    }


def _json_default(obj: Any) -> Any:
    """Fallback serialiser for types json.dumps can't handle natively."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)
