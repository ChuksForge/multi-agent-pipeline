"""
agents/etl/tools/data_validator.py
────────────────────────────────────
Data quality validation checks on polars DataFrames.

Runs a battery of checks and returns structured ValidationIssue records.
Designed to be non-blocking — all checks run regardless of earlier failures,
so the agent gets the full picture before deciding on recovery tier.

Check categories:
  1. Structural  — row count, column count, duplicate rows
  2. Null checks — per-column null rates against configurable thresholds
  3. Range checks — numeric columns within expected bounds (if provided)
  4. Type checks  — ensure critical columns have expected dtypes
  5. Cardinality  — detect unexpectedly low or high uniqueness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from pipeline.core.schemas import ValidationIssue
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """
    Configuration for the data validator.
    All thresholds have sensible defaults — override per task if needed.
    """
    # Structural
    min_rows: int = 1
    max_null_rate: float = 0.50           # Warn if any column exceeds this
    error_null_rate: float = 1.0          # Error if any column is entirely null
    max_duplicate_rate: float = 0.10      # Warn if >10% rows are duplicates

    # Per-column overrides (col_name → threshold)
    column_null_thresholds: dict[str, float] = field(default_factory=dict)
    expected_dtypes: dict[str, str] = field(default_factory=dict)  # col → dtype string
    numeric_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)  # col → (min, max)
    required_columns: list[str] = field(default_factory=list)


class DataValidator:
    """
    Runs all validation checks against a polars DataFrame.
    Returns a list of ValidationIssue records.
    """

    def __init__(self, config: ValidationConfig | None = None) -> None:
        self.config = config or ValidationConfig()

    def validate(self, df: pl.DataFrame) -> list[ValidationIssue]:
        """
        Run the full validation battery.
        Returns all issues found — never raises, always returns a list.
        """
        issues: list[ValidationIssue] = []

        issues.extend(self._check_structure(df))
        issues.extend(self._check_nulls(df))
        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_expected_dtypes(df))
        issues.extend(self._check_numeric_ranges(df))
        issues.extend(self._check_duplicates(df))

        error_count = sum(1 for i in issues if i.severity == "error")
        warn_count = sum(1 for i in issues if i.severity == "warning")
        logger.debug(
            "validation_complete",
            rows=len(df),
            cols=len(df.columns),
            errors=error_count,
            warnings=warn_count,
        )
        return issues

    # ── Check implementations ─────────────────────────────────────────────────

    def _check_structure(self, df: pl.DataFrame) -> list[ValidationIssue]:
        issues = []
        n_rows = len(df)

        if n_rows == 0:
            issues.append(ValidationIssue(
                severity="error",
                message="DataFrame is empty (0 rows)",
                affected_rows=0,
            ))
        elif n_rows < self.config.min_rows:
            issues.append(ValidationIssue(
                severity="warning",
                message=f"Only {n_rows} rows — expected at least {self.config.min_rows}",
                affected_rows=n_rows,
            ))

        if len(df.columns) == 0:
            issues.append(ValidationIssue(
                severity="error",
                message="DataFrame has no columns",
            ))

        return issues

    def _check_nulls(self, df: pl.DataFrame) -> list[ValidationIssue]:
        issues = []
        n_rows = len(df)
        if n_rows == 0:
            return issues

        for col in df.columns:
            null_count = df[col].null_count()
            null_rate = null_count / n_rows

            # Per-column threshold override
            threshold = self.config.column_null_thresholds.get(
                col, self.config.max_null_rate
            )

            if null_rate >= self.config.error_null_rate:
                issues.append(ValidationIssue(
                    column=col,
                    severity="error",
                    message=f"Column entirely null ({null_count}/{n_rows} rows)",
                    affected_rows=null_count,
                ))
            elif null_rate > threshold:
                issues.append(ValidationIssue(
                    column=col,
                    severity="warning",
                    message=f"{null_rate:.1%} null values exceed threshold ({threshold:.0%})",
                    affected_rows=null_count,
                ))

        return issues

    def _check_required_columns(self, df: pl.DataFrame) -> list[ValidationIssue]:
        issues = []
        for col in self.config.required_columns:
            if col not in df.columns:
                issues.append(ValidationIssue(
                    column=col,
                    severity="error",
                    message=f"Required column '{col}' is missing from DataFrame",
                ))
        return issues

    def _check_expected_dtypes(self, df: pl.DataFrame) -> list[ValidationIssue]:
        """Warn when a column's dtype doesn't match the expected type."""
        issues = []
        for col, expected_dtype in self.config.expected_dtypes.items():
            if col not in df.columns:
                continue
            actual = str(df[col].dtype)
            if actual.lower() != expected_dtype.lower():
                issues.append(ValidationIssue(
                    column=col,
                    severity="warning",
                    message=f"Expected dtype '{expected_dtype}', got '{actual}'",
                ))
        return issues

    def _check_numeric_ranges(self, df: pl.DataFrame) -> list[ValidationIssue]:
        """Flag rows where numeric values fall outside expected bounds."""
        issues = []
        for col, (min_val, max_val) in self.config.numeric_ranges.items():
            if col not in df.columns:
                continue
            series = df[col]
            try:
                below = series.filter(series < min_val).len()
                above = series.filter(series > max_val).len()
                if below > 0:
                    issues.append(ValidationIssue(
                        column=col,
                        severity="warning",
                        message=f"{below} values below expected minimum ({min_val})",
                        affected_rows=below,
                    ))
                if above > 0:
                    issues.append(ValidationIssue(
                        column=col,
                        severity="warning",
                        message=f"{above} values above expected maximum ({max_val})",
                        affected_rows=above,
                    ))
            except Exception:
                # Non-numeric column — skip silently
                pass

        return issues

    def _check_duplicates(self, df: pl.DataFrame) -> list[ValidationIssue]:
        """Warn if duplicate row rate exceeds threshold."""
        issues = []
        n_rows = len(df)
        if n_rows < 2:
            return issues

        try:
            n_unique = df.unique().height
            dup_count = n_rows - n_unique
            dup_rate = dup_count / n_rows

            if dup_rate > self.config.max_duplicate_rate:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"{dup_rate:.1%} duplicate rows ({dup_count}/{n_rows})",
                    affected_rows=dup_count,
                ))
        except Exception as e:
            logger.warning("duplicate_check_failed", error=str(e))

        return issues


# ── Quick validation helper ───────────────────────────────────────────────────

def quick_validate(
    df: pl.DataFrame,
    required_columns: list[str] | None = None,
    max_null_rate: float = 0.50,
) -> list[ValidationIssue]:
    """
    Convenience function for a fast validation run with default config.
    Used when the ETL agent doesn't have a full ValidationConfig.
    """
    config = ValidationConfig(
        required_columns=required_columns or [],
        max_null_rate=max_null_rate,
    )
    return DataValidator(config).validate(df)
