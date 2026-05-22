"""
agents/etl/agent.py
────────────────────
ETL Agent — LangGraph node.

Responsibilities:
  1. Register each DataSource with DuckDBTool
  2. Load data into a polars DataFrame via file_reader or DuckDB query
  3. Infer schema via schema_infer
  4. Validate via data_validator
  5. Apply type casts on suggestion
  6. Serialise result into ETLResult and write to PipelineState

Recovery tiers (set on ETLResult.recovery_tier):
  NONE       — clean run, no issues
  RETRY      — transient error, orchestrator should retry (not set here — set by orchestrator)
  SIMPLIFIED — loaded with row limit / sample only
  DEGRADED   — only schema returned, no data_json (partial failure)

The agent never raises into the graph. All exceptions are caught, logged,
and surfaced as AgentErrorRecord entries in PipelineState.errors.
"""

from __future__ import annotations

import time
from typing import Any

import polars as pl

from pipeline.agents.etl.prompts import (
    LOAD_TASK_TEMPLATE,
    SYSTEM_PROMPT,
    format_sources_block,
)
from pipeline.agents.etl.tools.data_validator import DataValidator, ValidationConfig
from pipeline.agents.etl.tools.duckdb_tool import DuckDBTool
from pipeline.agents.etl.tools.file_reader import read_file
from pipeline.agents.etl.tools.schema_infer import apply_casts, infer_schema, suggest_casts
from pipeline.core.exceptions import (
    DataSourceNotFoundError,
    ETLError,
    SchemaInferenceError,
    UnsupportedFileFormatError,
)
from pipeline.core.schemas import (
    AgentErrorRecord,
    DataSource,
    DataSourceType,
    ETLResult,
    PipelineStatus,
    RecoveryTier,
    TaskPlan,
)
from pipeline.core.state import PipelineState
from pipeline.middleware.logger import bind_pipeline_context, get_logger
from pipeline.middleware.token_tracker import make_tracker

logger = get_logger(__name__)

AGENT_ID = "etl"
_SAMPLE_ROWS_ON_FALLBACK = 5_000


# ─── LangGraph Node ──────────────────────────────────────────────────────────


def etl_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node function for the ETL Agent.

    Receives the full PipelineState, returns a partial update dict.
    Never raises — all errors are captured into the returned dict.

    Return keys updated: etl_result, errors, cost_log, status, current_agent
    """
    task_id = state["task_id"]
    bind_pipeline_context(task_id=task_id, agent_id=AGENT_ID)
    tracker = make_tracker(task_id=task_id, agent_id=AGENT_ID)
    start = time.monotonic()

    logger.info("etl_agent_start", task_id=task_id)

    task_plan: TaskPlan | None = state.get("task_plan")
    if task_plan is None:
        logger.error("etl_agent_no_task_plan")
        return _error_update(
            task_id=task_id,
            message="ETL agent received no TaskPlan — cannot proceed",
            error_type="MissingTaskPlan",
            recovery_tier=RecoveryTier.DEGRADED,
            elapsed=time.monotonic() - start,
            tracker_entries=tracker.entries,
        )

    sources = task_plan.data_sources
    if not sources:
        logger.warning("etl_agent_no_sources", note="Returning empty ETLResult")
        return _empty_result_update(task_id=task_id, elapsed=time.monotonic() - start)

    # ── Attempt full load ─────────────────────────────────────────────────────
    try:
        result = _run_etl(
            task_id=task_id,
            sources=sources,
            row_limit=task_plan.metadata.get("row_limit"),
        )
        elapsed = time.monotonic() - start
        result.elapsed_seconds = round(elapsed, 3)

        logger.info(
            "etl_agent_complete",
            rows=result.row_count,
            cols=result.column_count,
            issues=len(result.validation_issues),
            recovery_tier=result.recovery_tier.value,
            elapsed_s=round(elapsed, 2),
        )

        return {
            "etl_result": result,
            "cost_log": tracker.entries,
            "current_agent": None,
            "status": PipelineStatus.RUNNING,
        }

    # ── Tier 2: Simplified fallback (row-limited sample) ─────────────────────
    except (ETLError, DataSourceNotFoundError, UnsupportedFileFormatError) as e:
        logger.warning(
            "etl_agent_fallback",
            error=str(e),
            strategy="sample_only",
        )
        try:
            result = _run_etl(
                task_id=task_id,
                sources=sources,
                row_limit=_SAMPLE_ROWS_ON_FALLBACK,
                recovery_tier=RecoveryTier.SIMPLIFIED,
            )
            elapsed = time.monotonic() - start
            result.elapsed_seconds = round(elapsed, 3)
            result.warnings.append(
                f"Loaded as {_SAMPLE_ROWS_ON_FALLBACK}-row sample due to: {e}"
            )
            logger.info("etl_agent_simplified_complete", rows=result.row_count)
            return {
                "etl_result": result,
                "cost_log": tracker.entries,
                "current_agent": None,
                "status": PipelineStatus.PARTIAL,
            }

        # ── Tier 3: Schema-only degraded output ──────────────────────────────
        except Exception as fallback_err:
            logger.error(
                "etl_agent_degraded",
                error=str(fallback_err),
                strategy="schema_only",
            )
            error_record = AgentErrorRecord(
                agent_id=AGENT_ID,
                attempt=state.get("retry_counts", {}).get(AGENT_ID, 0) + 1,
                error_type=type(fallback_err).__name__,
                message=str(fallback_err),
                recovery_tier=RecoveryTier.DEGRADED,
            )
            degraded = ETLResult(
                task_id=task_id,
                source_ids=[s.source_id for s in sources],
                row_count=0,
                column_count=0,
                schema=[],
                recovery_tier=RecoveryTier.DEGRADED,
                warnings=[f"ETL fully degraded: {fallback_err}"],
            )
            return {
                "etl_result": degraded,
                "errors": [error_record],
                "cost_log": tracker.entries,
                "current_agent": None,
                "status": PipelineStatus.PARTIAL,
            }

    except Exception as unexpected:
        elapsed = time.monotonic() - start
        logger.exception("etl_agent_unexpected_error", error=str(unexpected))
        return _error_update(
            task_id=task_id,
            message=f"Unexpected ETL error: {unexpected}",
            error_type=type(unexpected).__name__,
            recovery_tier=RecoveryTier.DEGRADED,
            elapsed=elapsed,
            tracker_entries=tracker.entries,
        )


# ─── Core ETL Logic ───────────────────────────────────────────────────────────


def _run_etl(
    task_id: str,
    sources: list[DataSource],
    row_limit: int | None = None,
    recovery_tier: RecoveryTier = RecoveryTier.NONE,
) -> ETLResult:
    """
    Execute the full ETL pipeline for a list of sources.
    Returns ETLResult. May raise ETLError or subclasses on failure.
    """
    all_dfs: list[pl.DataFrame] = []
    source_ids: list[str] = []

    with DuckDBTool(row_limit=row_limit or 500_000) as db:
        for source in sources:
            df = _load_single_source(db, source, row_limit)
            all_dfs.append(df)
            source_ids.append(source.source_id)

    # Merge all sources (simple vertical concat if same schema, else side-by-side)
    df = _merge_dataframes(all_dfs)

    # Apply type cast suggestions
    casts = suggest_casts(df)
    if casts:
        logger.debug("applying_casts", columns=list(casts.keys()))
        df = apply_casts(df, casts)

    # Infer schema
    try:
        schemas, schema_issues = infer_schema(df)
    except Exception as e:
        raise SchemaInferenceError(source=str(sources[0].uri), detail=str(e)) from e

    # Validate
    validator = DataValidator(ValidationConfig())
    validation_issues = validator.validate(df)
    all_issues = schema_issues + validation_issues

    # Serialise DataFrame to JSON for cross-agent transfer
    data_json: str | None = None
    try:
        data_json = df.write_json()
    except Exception as e:
        logger.warning("dataframe_serialisation_failed", error=str(e))

    return ETLResult(
        task_id=task_id,
        source_ids=source_ids,
        row_count=len(df),
        column_count=len(df.columns),
        schema=schemas,
        validation_issues=all_issues,
        data_json=data_json,
        recovery_tier=recovery_tier,
    )


def _load_single_source(
    db: DuckDBTool,
    source: DataSource,
    row_limit: int | None,
) -> pl.DataFrame:
    """
    Load one DataSource into a polars DataFrame.

    Strategy:
      - For local files ≤ 50MB: use polars file_reader (faster for small files)
      - For Postgres / remote / large files: use DuckDB (lazy evaluation)
    """
    import os
    from pathlib import Path

    uri = source.uri
    is_remote = uri.startswith(("s3://", "http://", "https://", "postgresql://", "postgres://"))
    is_large = False

    if not is_remote:
        try:
            size = Path(uri).stat().st_size
            is_large = size > 50 * 1024 * 1024  # 50 MB threshold
        except OSError:
            pass

    if is_remote or is_large or source.source_type == DataSourceType.POSTGRES:
        # DuckDB path: lazy, handles remote and large files
        view_name = db.register_source(source)
        limit = row_limit or source.row_limit
        sql = f"SELECT * FROM {view_name}"
        if limit:
            sql += f" LIMIT {limit}"
        return db.execute(sql)
    else:
        # polars path: fast for small-medium local files
        return read_file(uri, row_limit=row_limit or source.row_limit)


def _merge_dataframes(dfs: list[pl.DataFrame]) -> pl.DataFrame:
    """
    Merge multiple DataFrames into one.

    Rules:
      - Single source: return as-is
      - Same columns: vertical concat (stack rows)
      - Different columns: horizontal concat with null-fill
    """
    if not dfs:
        return pl.DataFrame()
    if len(dfs) == 1:
        return dfs[0]

    # Check if all DataFrames have identical column sets
    col_sets = [set(df.columns) for df in dfs]
    if len(set(frozenset(c) for c in col_sets)) == 1:
        try:
            return pl.concat(dfs, how="vertical")
        except Exception:
            pass

    # Different schemas: horizontal concat (pad with nulls)
    try:
        max_rows = max(len(df) for df in dfs)
        padded = []
        for df in dfs:
            if len(df) < max_rows:
                padding = pl.DataFrame({col: [None] * (max_rows - len(df))
                                        for col in df.columns})
                df = pl.concat([df, padding])
            padded.append(df)
        return pl.concat(padded, how="horizontal")
    except Exception as e:
        logger.warning("merge_failed", error=str(e), note="Returning first DataFrame only")
        return dfs[0]


# ─── State update helpers ─────────────────────────────────────────────────────


def _error_update(
    task_id: str,
    message: str,
    error_type: str,
    recovery_tier: RecoveryTier,
    elapsed: float,
    tracker_entries: list,
) -> dict[str, Any]:
    """Build a partial state update dict for a failed ETL run."""
    error_record = AgentErrorRecord(
        agent_id=AGENT_ID,
        attempt=1,
        error_type=error_type,
        message=message,
        recovery_tier=recovery_tier,
    )
    degraded = ETLResult(
        task_id=task_id,
        source_ids=[],
        row_count=0,
        column_count=0,
        schema=[],
        recovery_tier=recovery_tier,
        elapsed_seconds=round(elapsed, 3),
        warnings=[message],
    )
    return {
        "etl_result": degraded,
        "errors": [error_record],
        "cost_log": tracker_entries,
        "current_agent": None,
        "status": PipelineStatus.PARTIAL,
    }


def _empty_result_update(task_id: str, elapsed: float) -> dict[str, Any]:
    """Return a valid empty ETLResult when no sources are provided."""
    return {
        "etl_result": ETLResult(
            task_id=task_id,
            source_ids=[],
            row_count=0,
            column_count=0,
            schema=[],
            elapsed_seconds=round(elapsed, 3),
        ),
        "cost_log": [],
        "current_agent": None,
        "status": PipelineStatus.RUNNING,
    }
