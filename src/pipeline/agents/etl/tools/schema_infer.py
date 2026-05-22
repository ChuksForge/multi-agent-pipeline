"""
agents/etl/tools/schema_infer.py
─────────────────────────────────
Schema inference from polars DataFrames.

Takes a raw polars DataFrame and produces a list of ColumnSchema objects
with dtype, nullability, null rate, cardinality, and sample values.
Also emits type-cast suggestions when polars chose a safe fallback type
(e.g. a numeric column read as String due to mixed values).
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pipeline.core.schemas import ColumnSchema, ValidationIssue
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

# Polars numeric dtype groups
_NUMERIC_TYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}
_DATE_TYPES = {pl.Date, pl.Datetime, pl.Duration, pl.Time}
_HIGH_CARDINALITY_THRESHOLD = 0.95  # >95% unique → probably an ID column


def infer_schema(df: pl.DataFrame) -> tuple[list[ColumnSchema], list[ValidationIssue]]:
    """
    Infer ColumnSchema for every column in the DataFrame.

    Returns:
        schemas: List of ColumnSchema, one per column.
        issues:  Validation warnings for type mismatches, high nulls, etc.
    """
    schemas: list[ColumnSchema] = []
    issues: list[ValidationIssue] = []
    n_rows = len(df)

    for col_name in df.columns:
        series = df[col_name]
        dtype = series.dtype
        null_count = series.null_count()
        null_rate = null_count / n_rows if n_rows > 0 else 0.0

        # Sample up to 5 non-null values
        sample_values = _sample_values(series, n=5)

        # Unique count (skip for very large string columns — too slow)
        unique_count: int | None = None
        if dtype != pl.Utf8 or n_rows <= 100_000:
            try:
                unique_count = series.n_unique()
            except Exception:
                pass

        schema = ColumnSchema(
            name=col_name,
            dtype=str(dtype),
            nullable=null_count > 0,
            null_rate=round(null_rate, 4),
            unique_count=unique_count,
            sample_values=sample_values,
        )
        schemas.append(schema)

        # ── Emit validation issues ──────────────────────────────────────────

        # High null rate warning
        if null_rate > 0.50:
            issues.append(ValidationIssue(
                column=col_name,
                severity="warning",
                message=f"{null_rate:.0%} null values ({null_count}/{n_rows} rows)",
                affected_rows=null_count,
            ))

        # String column that looks numeric (could be cast)
        if dtype == pl.Utf8 and n_rows > 0:
            non_null = series.drop_nulls()
            if len(non_null) > 0 and _looks_numeric(non_null):
                issues.append(ValidationIssue(
                    column=col_name,
                    severity="warning",
                    message=f"Column '{col_name}' is String but values appear numeric — consider casting",
                    affected_rows=0,
                ))

        # High-cardinality string column (likely ID — not useful for analysis)
        if dtype == pl.Utf8 and unique_count is not None and n_rows > 10:
            card_ratio = unique_count / n_rows
            if card_ratio > _HIGH_CARDINALITY_THRESHOLD:
                issues.append(ValidationIssue(
                    column=col_name,
                    severity="warning",
                    message=f"High-cardinality column ({card_ratio:.0%} unique) — likely an ID, may not be useful for analysis",
                    affected_rows=0,
                ))

        # Completely empty column
        if null_count == n_rows and n_rows > 0:
            issues.append(ValidationIssue(
                column=col_name,
                severity="error",
                message=f"Column '{col_name}' is entirely null",
                affected_rows=n_rows,
            ))

    logger.debug(
        "schema_inferred",
        columns=len(schemas),
        issues=len(issues),
        rows=n_rows,
    )
    return schemas, issues


def suggest_casts(df: pl.DataFrame) -> dict[str, str]:
    """
    Return a dict of {col_name: suggested_polars_dtype} for columns
    that could be cast to a more appropriate type.

    Used by the ETL agent to produce a cleaned DataFrame.
    """
    suggestions: dict[str, str] = {}
    n_rows = len(df)
    if n_rows == 0:
        return suggestions

    for col_name in df.columns:
        series = df[col_name]
        if series.dtype != pl.Utf8:
            continue

        non_null = series.drop_nulls()
        if len(non_null) == 0:
            continue

        if _looks_integer(non_null):
            suggestions[col_name] = "Int64"
        elif _looks_float(non_null):
            suggestions[col_name] = "Float64"
        elif _looks_boolean(non_null):
            suggestions[col_name] = "Boolean"
        elif _looks_date(non_null):
            suggestions[col_name] = "Date"

    return suggestions


def apply_casts(df: pl.DataFrame, casts: dict[str, str]) -> pl.DataFrame:
    """
    Apply a dict of {col_name: polars_dtype_str} to a DataFrame.
    Silently skips columns that fail to cast (best-effort).
    """
    _DTYPE_MAP: dict[str, Any] = {
        "Int64": pl.Int64,
        "Float64": pl.Float64,
        "Boolean": pl.Boolean,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Utf8": pl.Utf8,
        "String": pl.Utf8,
    }

    expressions = []
    for col_name, dtype_str in casts.items():
        if col_name not in df.columns:
            continue
        target_dtype = _DTYPE_MAP.get(dtype_str)
        if target_dtype is None:
            continue
        try:
            expressions.append(
                pl.col(col_name).cast(target_dtype, strict=False).alias(col_name)
            )
        except Exception as e:
            logger.warning("cast_failed", column=col_name, target=dtype_str, error=str(e))

    if expressions:
        df = df.with_columns(expressions)

    return df


# ── Private helpers ───────────────────────────────────────────────────────────

def _sample_values(series: pl.Series, n: int = 5) -> list[Any]:
    """Return up to n non-null sample values from a series."""
    try:
        non_null = series.drop_nulls()
        sample = non_null.head(n)
        return [v for v in sample.to_list() if v is not None]
    except Exception:
        return []


def _looks_numeric(series: pl.Series) -> bool:
    """True if ≥90% of non-null values parse as float."""
    try:
        parsed = series.cast(pl.Float64, strict=False)
        non_null_parsed = parsed.drop_nulls()
        return len(non_null_parsed) / max(len(series), 1) >= 0.90
    except Exception:
        return False


def _looks_integer(series: pl.Series) -> bool:
    """True if values look like integers (no decimal point)."""
    try:
        sample = series.head(200).to_list()
        numeric = [s for s in sample if s and str(s).strip().lstrip("-").isdigit()]
        return len(numeric) / max(len(sample), 1) >= 0.90
    except Exception:
        return False


def _looks_float(series: pl.Series) -> bool:
    """True if values look like floating-point numbers."""
    try:
        parsed = series.cast(pl.Float64, strict=False)
        success_rate = len(parsed.drop_nulls()) / max(len(series), 1)
        return success_rate >= 0.90
    except Exception:
        return False


def _looks_boolean(series: pl.Series) -> bool:
    """True if values are in a recognised boolean value set."""
    BOOL_VALUES = {
        "true", "false", "yes", "no", "1", "0",
        "t", "f", "y", "n", "on", "off",
    }
    try:
        sample = {str(v).strip().lower() for v in series.head(50).to_list() if v}
        return sample.issubset(BOOL_VALUES) and len(sample) >= 1
    except Exception:
        return False


def _looks_date(series: pl.Series) -> bool:
    """True if polars can parse values as dates."""
    try:
        parsed = series.str.to_date(strict=False)
        success_rate = len(parsed.drop_nulls()) / max(len(series), 1)
        return success_rate >= 0.80
    except Exception:
        return False
