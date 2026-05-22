"""
agents/analysis/tools/stats_engine.py
──────────────────────────────────────
Descriptive statistics engine using polars.

Produces one SummaryStats record per column. Numeric columns get full
quantile/mean/std coverage. Non-numeric columns get cardinality and
top-value distributions. All computation is deterministic — no LLM involved.

Design:
  - polars.DataFrame.describe() used as base, then extended manually
    for percentiles and top-values (describe() alone is not enough)
  - Non-numeric columns handled separately — no mean/std attempted
  - Entirely null columns produce a minimal record with null_count = n_rows
  - Returns empty list on empty DataFrame (not an error)
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pipeline.core.schemas import SummaryStats
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

_NUMERIC_DTYPES = (
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
)


def compute_stats(df: pl.DataFrame) -> list[SummaryStats]:
    """
    Compute descriptive statistics for every column in df.

    Returns:
        List of SummaryStats, one per column.
        Empty list if df has no rows or no columns.
    """
    if df.is_empty() or len(df.columns) == 0:
        return []

    stats: list[SummaryStats] = []
    n_rows = len(df)

    for col_name in df.columns:
        series = df[col_name]
        dtype = series.dtype
        null_count = series.null_count()
        non_null = series.drop_nulls()
        count = len(non_null)

        if dtype in _NUMERIC_DTYPES:
            record = _numeric_stats(col_name, dtype, n_rows, null_count, non_null)
        else:
            record = _categorical_stats(col_name, dtype, n_rows, null_count, non_null)

        stats.append(record)

    logger.debug(
        "stats_computed",
        columns=len(stats),
        rows=n_rows,
    )
    return stats


def compute_numeric_stats_only(df: pl.DataFrame) -> list[SummaryStats]:
    """
    Compute stats for numeric columns only.
    Used by anomaly_detector when it needs to know which columns to fit on.
    """
    numeric_cols = [
        c for c in df.columns
        if df[c].dtype in _NUMERIC_DTYPES
    ]
    if not numeric_cols:
        return []
    return compute_stats(df.select(numeric_cols))


def numeric_column_names(df: pl.DataFrame) -> list[str]:
    """Return names of all numeric columns in df."""
    return [c for c in df.columns if df[c].dtype in _NUMERIC_DTYPES]


# ── Private helpers ───────────────────────────────────────────────────────────


def _numeric_stats(
    col_name: str,
    dtype: pl.PolarsDataType,
    n_rows: int,
    null_count: int,
    non_null: pl.Series,
) -> SummaryStats:
    """Full descriptive stats for a numeric column."""
    count = len(non_null)

    if count == 0:
        return SummaryStats(
            column=col_name,
            dtype=str(dtype),
            count=0,
            null_count=null_count,
        )

    # Cast to Float64 for consistent arithmetic
    s = non_null.cast(pl.Float64)

    try:
        mean = float(s.mean())
        std = float(s.std()) if count > 1 else 0.0
        min_val = float(s.min())
        max_val = float(s.max())
        p25 = float(s.quantile(0.25, interpolation="linear"))
        p50 = float(s.quantile(0.50, interpolation="linear"))
        p75 = float(s.quantile(0.75, interpolation="linear"))
    except Exception as e:
        logger.warning("numeric_stats_partial_failure", column=col_name, error=str(e))
        mean = std = min_val = max_val = p25 = p50 = p75 = None  # type: ignore[assignment]

    return SummaryStats(
        column=col_name,
        dtype=str(dtype),
        count=count,
        null_count=null_count,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
        p25=p25,
        p50=p50,
        p75=p75,
    )


def _categorical_stats(
    col_name: str,
    dtype: pl.PolarsDataType,
    n_rows: int,
    null_count: int,
    non_null: pl.Series,
) -> SummaryStats:
    """Cardinality and top-values for non-numeric columns."""
    count = len(non_null)

    top_values: list[Any] = []
    try:
        # Top 5 most frequent values
        vc = non_null.value_counts(sort=True).head(5)
        top_values = vc[col_name].to_list()
    except Exception:
        try:
            top_values = non_null.head(5).to_list()
        except Exception:
            top_values = []

    return SummaryStats(
        column=col_name,
        dtype=str(dtype),
        count=count,
        null_count=null_count,
        top_values=top_values,
    )
