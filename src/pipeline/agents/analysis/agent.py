"""
agents/analysis/agent.py
─────────────────────────
Analysis Agent — LangGraph node.

Responsibilities:
  1. Deserialise ETLResult.data_json back to a polars DataFrame
  2. Compute summary statistics via stats_engine
  3. Detect anomalies via anomaly_detector (IsolationForest + Z-score)
  4. Build charts via chart_builder (auto-selected Vega-Lite specs)
  5. Generate key_findings via one small LLM call (Haiku)
  6. Write AnalysisResult into PipelineState

Recovery tiers:
  NONE       — full analysis, all methods ran
  SIMPLIFIED — IsolationForest sampled due to large data, or Z-score only
  DEGRADED   — stats only, no anomaly detection (InsufficientDataError)
  DEGRADED   — no data available from ETL (empty data_json / zero rows)

The agent never raises into the graph.
One LLM call only — for key_findings generation.
All numbers come from tool outputs.
"""

from __future__ import annotations

import json
import time
from typing import Any

import polars as pl
from langchain_anthropic import ChatAnthropic

from pipeline.agents.analysis.prompts import (
    KEY_FINDINGS_TEMPLATE,
    SYSTEM_PROMPT,
    format_anomalies_block,
    format_issues_block,
    format_stats_block,
)
from pipeline.agents.analysis.tools.anomaly_detector import (
    AnomalyConfig,
    detect_anomalies,
)
from pipeline.agents.analysis.tools.chart_builder import auto_select_charts
from pipeline.agents.analysis.tools.stats_engine import compute_stats
from pipeline.core.config import settings
from pipeline.core.exceptions import (
    AnalysisError,
    InsufficientDataError,
)
from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    AnomalyRecord,
    ETLResult,
    PipelineStatus,
    RecoveryTier,
    SummaryStats,
)
from pipeline.core.state import PipelineState
from pipeline.middleware.logger import bind_pipeline_context, get_logger
from pipeline.middleware.token_tracker import make_tracker

logger = get_logger(__name__)

AGENT_ID = "analysis"
_MIN_ROWS_FOR_ANALYSIS = 3


# ─── LangGraph Node ───────────────────────────────────────────────────────────


def analysis_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node function for the Analysis Agent.
    Receives full PipelineState, returns partial update dict.
    Never raises — all errors are captured in the return dict.
    """
    task_id = state["task_id"]
    bind_pipeline_context(task_id=task_id, agent_id=AGENT_ID)
    tracker = make_tracker(task_id=task_id, agent_id=AGENT_ID)
    start = time.monotonic()

    logger.info("analysis_agent_start", task_id=task_id)

    # ── Guard: need ETL result ────────────────────────────────────────────────
    etl_result: ETLResult | None = state.get("etl_result")
    if etl_result is None or etl_result.row_count == 0:
        logger.warning("analysis_agent_no_etl_data")
        return _degraded_update(
            task_id=task_id,
            message="No ETL data available for analysis",
            elapsed=time.monotonic() - start,
            tracker_entries=tracker.entries,
        )

    # ── Deserialise DataFrame ─────────────────────────────────────────────────
    df = _deserialise_dataframe(etl_result)
    if df is None or df.is_empty():
        return _degraded_update(
            task_id=task_id,
            message="Could not deserialise ETL data_json to DataFrame",
            elapsed=time.monotonic() - start,
            tracker_entries=tracker.entries,
        )

    # ── Attempt full analysis ─────────────────────────────────────────────────
    try:
        result = _run_analysis(
            task_id=task_id,
            df=df,
            etl_result=etl_result,
            tracker=tracker,
        )
        elapsed = time.monotonic() - start
        result.elapsed_seconds = round(elapsed, 3)

        logger.info(
            "analysis_agent_complete",
            rows=etl_result.row_count,
            anomalies=result.anomaly_count,
            charts=len(result.charts),
            findings=len(result.key_findings),
            recovery_tier=result.recovery_tier.value,
            elapsed_s=round(elapsed, 2),
        )
        return {
            "analysis_result": result,
            "cost_log": tracker.entries,
            "current_agent": None,
            "status": PipelineStatus.RUNNING,
        }

    # ── Tier 2: stats only, skip anomaly detection ────────────────────────────
    except InsufficientDataError as e:
        logger.warning("analysis_agent_insufficient_data", error=str(e))
        try:
            result = _run_stats_only(
                task_id=task_id,
                df=df,
                etl_result=etl_result,
                tracker=tracker,
            )
            elapsed = time.monotonic() - start
            result.elapsed_seconds = round(elapsed, 3)
            result.warnings.append(f"Anomaly detection skipped: {e}")
            return {
                "analysis_result": result,
                "cost_log": tracker.entries,
                "current_agent": None,
                "status": PipelineStatus.PARTIAL,
            }
        except Exception as fallback_err:
            return _degraded_update(
                task_id=task_id,
                message=f"Stats-only fallback failed: {fallback_err}",
                elapsed=time.monotonic() - start,
                tracker_entries=tracker.entries,
                attempt=state.get("retry_counts", {}).get(AGENT_ID, 0) + 1,
            )

    except Exception as unexpected:
        logger.exception("analysis_agent_unexpected_error", error=str(unexpected))
        return _degraded_update(
            task_id=task_id,
            message=f"Unexpected analysis error: {unexpected}",
            elapsed=time.monotonic() - start,
            tracker_entries=tracker.entries,
            attempt=state.get("retry_counts", {}).get(AGENT_ID, 0) + 1,
        )


# ─── Core Analysis Logic ──────────────────────────────────────────────────────


def _run_analysis(
    task_id: str,
    df: pl.DataFrame,
    etl_result: ETLResult,
    tracker: Any,
) -> AnalysisResult:
    """Full analysis: stats + anomaly detection + charts + LLM findings."""

    # 1. Summary statistics
    summary_stats = compute_stats(df)

    # 2. Anomaly detection
    anomaly_cfg = AnomalyConfig()
    anomalies, recovery_tier = detect_anomalies(df, config=anomaly_cfg)

    # 3. Charts (auto-selected)
    charts = []
    try:
        charts = auto_select_charts(df, anomalies, max_charts=4)
    except Exception as e:
        logger.warning("chart_building_failed", error=str(e))

    # 4. Key findings via LLM (one Haiku call)
    key_findings = _generate_key_findings(
        df=df,
        summary_stats=summary_stats,
        anomalies=anomalies,
        etl_result=etl_result,
        tracker=tracker,
    )

    n_rows = len(df)
    return AnalysisResult(
        task_id=task_id,
        summary_stats=summary_stats,
        anomalies=anomalies,
        anomaly_rate=round(len(anomalies) / n_rows, 4) if n_rows > 0 else 0.0,
        charts=charts,
        key_findings=key_findings,
        recovery_tier=recovery_tier,
    )


def _run_stats_only(
    task_id: str,
    df: pl.DataFrame,
    etl_result: ETLResult,
    tracker: Any,
) -> AnalysisResult:
    """Tier 2: stats only — anomaly detection skipped."""
    summary_stats = compute_stats(df)

    # Still attempt charts and findings even without anomalies
    charts = []
    try:
        charts = auto_select_charts(df, anomalies=[], max_charts=2)
    except Exception:
        pass

    key_findings = _generate_key_findings(
        df=df,
        summary_stats=summary_stats,
        anomalies=[],
        etl_result=etl_result,
        tracker=tracker,
    )

    return AnalysisResult(
        task_id=task_id,
        summary_stats=summary_stats,
        anomalies=[],
        anomaly_rate=0.0,
        charts=charts,
        key_findings=key_findings,
        recovery_tier=RecoveryTier.DEGRADED,
    )


def _generate_key_findings(
    df: pl.DataFrame,
    summary_stats: list[SummaryStats],
    anomalies: list[AnomalyRecord],
    etl_result: ETLResult,
    tracker: Any,
) -> list[str]:
    """
    One LLM call to generate key findings from structured data.
    Returns empty list on failure (non-blocking).
    """
    try:
        stats_dicts = [s.model_dump() for s in summary_stats]
        issues_dicts = [i.model_dump() for i in etl_result.validation_issues]
        anomaly_dicts = [a.model_dump() for a in anomalies[:5]]

        detection_method = "ensemble (IsolationForest + Z-score)"
        if not anomalies:
            detection_method = "ensemble (no anomalies found)"

        prompt = KEY_FINDINGS_TEMPLATE.format(
            row_count=len(df),
            col_count=len(df.columns),
            anomaly_count=len(anomalies),
            anomaly_rate=len(anomalies) / max(len(df), 1),
            detection_method=detection_method,
            stats_block=format_stats_block(stats_dicts),
            issues_block=format_issues_block(issues_dicts),
            anomalies_block=format_anomalies_block(anomaly_dicts),
        )

        llm = ChatAnthropic(
            model=settings.analysis_agent_model,
            max_tokens=512,
            callbacks=[tracker],
        )
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )

        # Strip markdown fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        findings = json.loads(content)
        if isinstance(findings, list):
            return [str(f) for f in findings if f]
        return []

    except Exception as e:
        logger.warning("key_findings_llm_failed", error=str(e))
        # Graceful fallback: generate findings from structured data directly
        return _fallback_findings(summary_stats, anomalies, len(df))


def _fallback_findings(
    summary_stats: list[SummaryStats],
    anomalies: list[AnomalyRecord],
    n_rows: int,
) -> list[str]:
    """
    Deterministic findings when the LLM call fails.
    Based purely on structured data — always safe to call.
    """
    findings = []
    findings.append(f"Dataset contains {n_rows:,} rows across {len(summary_stats)} columns.")

    if anomalies:
        findings.append(
            f"{len(anomalies)} anomalies detected "
            f"({len(anomalies)/max(n_rows,1):.1%} of rows)."
        )
        top = anomalies[0]
        findings.append(
            f"Most significant anomaly at row {top.row_index}: "
            f"{top.column} = {top.value} (score={top.anomaly_score:.3f})."
        )
    else:
        findings.append("No anomalies detected in the dataset.")

    # High-null columns
    high_null = [s for s in summary_stats if s.null_count > 0 and s.count > 0
                 and s.null_count / (s.count + s.null_count) > 0.3]
    if high_null:
        col_names = ", ".join(s.column for s in high_null[:3])
        findings.append(f"Columns with >30% null values: {col_names}.")

    return findings


# ─── DataFrame deserialisation ────────────────────────────────────────────────


def _deserialise_dataframe(etl_result: ETLResult) -> pl.DataFrame | None:
    """
    Reconstruct a polars DataFrame from ETLResult.data_json.
    Returns None if data_json is absent or unparseable.
    """
    if not etl_result.data_json:
        logger.warning("analysis_no_data_json")
        return None
    try:
        import io
        df = pl.read_json(io.StringIO(etl_result.data_json))
        return df
    except Exception as e:
        logger.warning("dataframe_deserialise_failed", error=str(e))
        return None


# ─── State update helpers ─────────────────────────────────────────────────────


def _degraded_update(
    task_id: str,
    message: str,
    elapsed: float,
    tracker_entries: list,
    attempt: int = 1,
) -> dict[str, Any]:
    """Return a partial state update for a fully degraded analysis run."""
    error_record = AgentErrorRecord(
        agent_id=AGENT_ID,
        attempt=attempt,
        error_type="AnalysisDegraded",
        message=message,
        recovery_tier=RecoveryTier.DEGRADED,
    )
    degraded = AnalysisResult(
        task_id=task_id,
        recovery_tier=RecoveryTier.DEGRADED,
        elapsed_seconds=round(elapsed, 3),
        warnings=[message],
    )
    return {
        "analysis_result": degraded,
        "errors": [error_record],
        "cost_log": tracker_entries,
        "current_agent": None,
        "status": PipelineStatus.PARTIAL,
    }
