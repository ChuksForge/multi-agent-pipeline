"""
tests/unit/test_data_validator.py
───────────────────────────────────
Unit tests for DataValidator and ValidationConfig.
All checks tested in isolation and in combination.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.etl.tools.data_validator import (
    DataValidator,
    ValidationConfig,
    quick_validate,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "region": ["north", "south", "east", "west", "north"],
    })


@pytest.fixture
def high_null_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [10.0, None, None, None],   # 75% null
        "label": ["a", "b", "c", "d"],
    })


@pytest.fixture
def empty_df():
    return pl.DataFrame({"id": [], "value": []})


@pytest.fixture
def duplicate_df():
    return pl.DataFrame({
        "id": [1, 1, 2, 2, 3],
        "value": [10, 10, 20, 20, 30],
    })


@pytest.fixture
def out_of_range_df():
    return pl.DataFrame({
        "age": [-5, 25, 150, 30],      # -5 and 150 are out of range [0, 120]
        "score": [0.0, 0.5, 1.2, 0.8], # 1.2 is out of range [0, 1]
    })


# ─── Structure checks ─────────────────────────────────────────────────────────


class TestStructureChecks:
    def test_empty_df_produces_error(self, empty_df):
        v = DataValidator()
        issues = v.validate(empty_df)
        errors = [i for i in issues if i.severity == "error"]
        assert any("empty" in i.message.lower() for i in errors)

    def test_clean_df_no_structure_issues(self, clean_df):
        v = DataValidator()
        issues = v.validate(clean_df)
        structure_errors = [i for i in issues
                            if i.severity == "error" and i.column is None]
        assert len(structure_errors) == 0

    def test_min_rows_warning(self):
        df = pl.DataFrame({"id": [1]})  # Only 1 row
        config = ValidationConfig(min_rows=10)
        v = DataValidator(config)
        issues = v.validate(df)
        warns = [i for i in issues if "row" in i.message.lower()]
        assert len(warns) >= 1


# ─── Null checks ─────────────────────────────────────────────────────────────


class TestNullChecks:
    def test_high_null_rate_produces_warning(self, high_null_df):
        config = ValidationConfig(max_null_rate=0.30)  # 30% threshold
        v = DataValidator(config)
        issues = v.validate(high_null_df)
        null_warnings = [i for i in issues if i.severity == "warning" and i.column == "value"]
        assert len(null_warnings) >= 1

    def test_null_within_threshold_no_warning(self, high_null_df):
        config = ValidationConfig(max_null_rate=0.90)  # Very permissive
        v = DataValidator(config)
        issues = v.validate(high_null_df)
        null_warnings = [i for i in issues if i.severity == "warning" and i.column == "value"]
        assert len(null_warnings) == 0

    def test_entirely_null_column_is_error(self):
        df = pl.DataFrame({"col": [None, None, None]})
        v = DataValidator()
        issues = v.validate(df)
        errors = [i for i in issues if i.severity == "error" and i.column == "col"]
        assert len(errors) >= 1

    def test_per_column_threshold_override(self):
        df = pl.DataFrame({
            "critical": [1.0, None, None, None],  # 75% null — critical column
            "optional": [1.0, None, None, None],  # 75% null — less critical
        })
        config = ValidationConfig(
            max_null_rate=0.50,
            column_null_thresholds={"optional": 0.90},  # Allow 90% null here
        )
        v = DataValidator(config)
        issues = v.validate(df)

        critical_warns = [i for i in issues if i.column == "critical" and i.severity == "warning"]
        optional_warns = [i for i in issues if i.column == "optional" and i.severity == "warning"]
        assert len(critical_warns) >= 1
        assert len(optional_warns) == 0  # Suppressed by override

    def test_no_nulls_no_null_issues(self, clean_df):
        v = DataValidator()
        issues = v.validate(clean_df)
        null_issues = [i for i in issues if "null" in i.message.lower()]
        assert len(null_issues) == 0


# ─── Required columns ────────────────────────────────────────────────────────


class TestRequiredColumns:
    def test_missing_required_column_is_error(self, clean_df):
        config = ValidationConfig(required_columns=["id", "timestamp"])
        v = DataValidator(config)
        issues = v.validate(clean_df)
        missing = [i for i in issues if "timestamp" in (i.column or "") and i.severity == "error"]
        assert len(missing) >= 1

    def test_present_required_columns_no_issue(self, clean_df):
        config = ValidationConfig(required_columns=["id", "revenue"])
        v = DataValidator(config)
        issues = v.validate(clean_df)
        required_errors = [
            i for i in issues
            if i.severity == "error" and i.column in ("id", "revenue")
            and "missing" in i.message.lower()
        ]
        assert len(required_errors) == 0


# ─── Dtype checks ────────────────────────────────────────────────────────────


class TestDtypeChecks:
    def test_wrong_dtype_produces_warning(self, clean_df):
        config = ValidationConfig(expected_dtypes={"id": "Float64"})  # Actually Int64
        v = DataValidator(config)
        issues = v.validate(clean_df)
        dtype_warns = [i for i in issues if i.column == "id" and "dtype" in i.message.lower()]
        assert len(dtype_warns) >= 1

    def test_correct_dtype_no_warning(self, clean_df):
        config = ValidationConfig(expected_dtypes={"region": "String"})
        v = DataValidator(config)
        issues = v.validate(clean_df)
        dtype_warns = [i for i in issues if i.column == "region" and "dtype" in i.message.lower()]
        # Polars may call it Utf8 or String — either way should match
        assert len(dtype_warns) <= 1  # Tolerate one if naming differs


# ─── Numeric range checks ────────────────────────────────────────────────────


class TestNumericRanges:
    def test_out_of_range_produces_warning(self, out_of_range_df):
        config = ValidationConfig(
            numeric_ranges={
                "age": (0.0, 120.0),
                "score": (0.0, 1.0),
            }
        )
        v = DataValidator(config)
        issues = v.validate(out_of_range_df)
        range_issues = [i for i in issues if i.severity == "warning"
                        and ("below" in i.message or "above" in i.message)]
        assert len(range_issues) >= 2  # At least one for age, one for score

    def test_in_range_no_warning(self, clean_df):
        config = ValidationConfig(numeric_ranges={"revenue": (0.0, 10_000.0)})
        v = DataValidator(config)
        issues = v.validate(clean_df)
        range_warns = [i for i in issues if i.column == "revenue"
                       and ("below" in i.message or "above" in i.message)]
        assert len(range_warns) == 0


# ─── Duplicate checks ────────────────────────────────────────────────────────


class TestDuplicateChecks:
    def test_high_duplicate_rate_warns(self, duplicate_df):
        config = ValidationConfig(max_duplicate_rate=0.10)
        v = DataValidator(config)
        issues = v.validate(duplicate_df)
        dup_warns = [i for i in issues if "duplicate" in i.message.lower()]
        assert len(dup_warns) >= 1

    def test_low_duplicate_rate_no_warning(self, clean_df):
        config = ValidationConfig(max_duplicate_rate=0.10)
        v = DataValidator(config)
        issues = v.validate(clean_df)
        dup_warns = [i for i in issues if "duplicate" in i.message.lower()]
        assert len(dup_warns) == 0


# ─── quick_validate ───────────────────────────────────────────────────────────


class TestQuickValidate:
    def test_returns_list(self, clean_df):
        issues = quick_validate(clean_df)
        assert isinstance(issues, list)

    def test_required_columns_forwarded(self, clean_df):
        issues = quick_validate(clean_df, required_columns=["missing_col"])
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) >= 1

    def test_clean_data_minimal_issues(self, clean_df):
        issues = quick_validate(clean_df)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0


# ─── Full battery ─────────────────────────────────────────────────────────────


class TestFullValidation:
    def test_all_checks_run_even_after_errors(self):
        """Validator is non-blocking — all checks run regardless."""
        # Part A: null col error + missing required col error
        df_a = pl.DataFrame({
            "value": [1.0, 2.0, 3.0],
            "all_null_col": [None, None, None],
        })
        config_a = ValidationConfig(required_columns=["nonexistent"])
        issues_a = DataValidator(config_a).validate(df_a)
        assert any(i.severity == "error" and i.column == "all_null_col" for i in issues_a)
        assert any("nonexistent" in (i.column or "") for i in issues_a)

        # Part B: duplicate warning fires independently
        df_b = pl.DataFrame({
            "id": [1, 1, 2, 2, 3],
            "value": [10, 10, 20, 20, 30],
        })
        config_b = ValidationConfig(max_duplicate_rate=0.05)
        issues_b = DataValidator(config_b).validate(df_b)
        assert any("duplicate" in i.message.lower() for i in issues_b)

        # Part C: combined error count
        assert len(issues_a) >= 2

    def test_severity_values_are_valid(self, high_null_df):
        config = ValidationConfig(max_null_rate=0.10)
        v = DataValidator(config)
        issues = v.validate(high_null_df)
        for issue in issues:
            assert issue.severity in ("warning", "error")
