"""
tests/unit/test_schema_infer.py
─────────────────────────────────
Unit tests for schema_infer: infer_schema, suggest_casts, apply_casts.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.etl.tools.schema_infer import apply_casts, infer_schema, suggest_casts
from pipeline.core.schemas import ColumnSchema


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0],
        "region": ["north", "south", "east", "west", "north"],
        "active": [True, False, True, True, False],
    })


@pytest.fixture
def nulls_df():
    return pl.DataFrame({
        "id": [1, 2, 3, 4],
        "value": [10.0, None, None, None],    # 75% null — clearly exceeds 50% threshold
        "label": ["a", None, None, "d"],
        "all_null": [None, None, None, None],
    })


@pytest.fixture
def string_numerics_df():
    """DataFrame where numbers were read as strings — needs cast suggestions."""
    return pl.DataFrame({
        "id": ["1", "2", "3", "4", "5"],
        "price": ["9.99", "19.99", "4.50", "29.00", "14.99"],
        "quantity": ["10", "5", "20", "2", "8"],
        "name": ["apple", "banana", "cherry", "date", "elderberry"],
    })


@pytest.fixture
def high_cardinality_df():
    return pl.DataFrame({
        "uuid": [f"id-{i:05d}" for i in range(100)],
        "value": list(range(100)),
    })


# ─── infer_schema ─────────────────────────────────────────────────────────────


class TestInferSchema:
    def test_returns_correct_column_count(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        assert len(schemas) == 4

    def test_column_names_preserved(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        names = {s.name for s in schemas}
        assert names == {"id", "revenue", "region", "active"}

    def test_dtype_populated(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        schema_map = {s.name: s for s in schemas}
        assert "Float" in schema_map["revenue"].dtype or "f64" in schema_map["revenue"].dtype.lower()

    def test_null_rate_zero_for_clean_data(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        for s in schemas:
            assert s.null_rate == pytest.approx(0.0)

    def test_null_rate_correct(self, nulls_df):
        schemas, _ = infer_schema(nulls_df)
        schema_map = {s.name: s for s in schemas}
        # value has 3 nulls out of 4 rows = 0.75
        assert schema_map["value"].null_rate == pytest.approx(0.75)

    def test_nullable_true_when_nulls_present(self, nulls_df):
        schemas, _ = infer_schema(nulls_df)
        schema_map = {s.name: s for s in schemas}
        assert schema_map["value"].nullable is True

    def test_nullable_false_when_no_nulls(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        schema_map = {s.name: s for s in schemas}
        assert schema_map["id"].nullable is False

    def test_sample_values_populated(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        schema_map = {s.name: s for s in schemas}
        assert len(schema_map["region"].sample_values) > 0
        assert len(schema_map["region"].sample_values) <= 5

    def test_unique_count_populated(self, clean_df):
        schemas, _ = infer_schema(clean_df)
        schema_map = {s.name: s for s in schemas}
        assert schema_map["region"].unique_count == 4  # north, south, east, west

    def test_entirely_null_column_raises_error_issue(self, nulls_df):
        _, issues = infer_schema(nulls_df)
        error_issues = [i for i in issues if i.severity == "error" and i.column == "all_null"]
        assert len(error_issues) == 1

    def test_high_null_rate_raises_warning(self, nulls_df):
        _, issues = infer_schema(nulls_df)
        # value column has 50% nulls — should warn
        warn_issues = [i for i in issues if i.severity == "warning" and i.column == "value"]
        assert len(warn_issues) >= 1

    def test_high_cardinality_warning(self, high_cardinality_df):
        _, issues = infer_schema(high_cardinality_df)
        cardinality_warns = [i for i in issues if "cardinality" in i.message.lower() or "unique" in i.message.lower()]
        assert len(cardinality_warns) >= 1

    def test_empty_dataframe(self):
        df = pl.DataFrame()
        schemas, issues = infer_schema(df)
        assert schemas == []
        assert issues == []


# ─── suggest_casts ────────────────────────────────────────────────────────────


class TestSuggestCasts:
    def test_numeric_strings_get_cast_suggestions(self, string_numerics_df):
        suggestions = suggest_casts(string_numerics_df)
        # price and quantity look numeric — should suggest Float64 or Int64
        assert "price" in suggestions or "quantity" in suggestions

    def test_non_numeric_strings_not_suggested(self, string_numerics_df):
        suggestions = suggest_casts(string_numerics_df)
        assert "name" not in suggestions

    def test_already_numeric_columns_not_suggested(self, clean_df):
        suggestions = suggest_casts(clean_df)
        # revenue is already Float64 — no cast suggestion needed
        assert "revenue" not in suggestions

    def test_empty_df_returns_empty(self):
        df = pl.DataFrame()
        suggestions = suggest_casts(df)
        assert suggestions == {}

    def test_boolean_string_detected(self):
        df = pl.DataFrame({"flag": ["true", "false", "true", "false"]})
        suggestions = suggest_casts(df)
        assert "flag" in suggestions
        assert suggestions["flag"] == "Boolean"


# ─── apply_casts ──────────────────────────────────────────────────────────────


class TestApplyCasts:
    def test_string_to_int64(self):
        df = pl.DataFrame({"qty": ["1", "2", "3"]})
        result = apply_casts(df, {"qty": "Int64"})
        assert result["qty"].dtype == pl.Int64

    def test_string_to_float64(self):
        df = pl.DataFrame({"price": ["9.99", "4.50", "12.00"]})
        result = apply_casts(df, {"price": "Float64"})
        assert result["price"].dtype == pl.Float64

    def test_failed_cast_does_not_raise(self):
        df = pl.DataFrame({"col": ["abc", "def", "ghi"]})
        # Trying to cast text to Int64 should not raise — silently skip
        result = apply_casts(df, {"col": "Int64"})
        assert isinstance(result, pl.DataFrame)

    def test_missing_column_skipped(self, clean_df):
        result = apply_casts(clean_df, {"nonexistent_col": "Float64"})
        assert result.columns == clean_df.columns

    def test_unknown_dtype_skipped(self, clean_df):
        result = apply_casts(clean_df, {"revenue": "FancyType123"})
        assert isinstance(result, pl.DataFrame)

    def test_multiple_casts_applied(self, string_numerics_df):
        casts = {"id": "Int64", "price": "Float64"}
        result = apply_casts(string_numerics_df, casts)
        assert result["id"].dtype == pl.Int64
        assert result["price"].dtype == pl.Float64

    def test_original_df_not_mutated(self, string_numerics_df):
        original_dtypes = {c: str(string_numerics_df[c].dtype) for c in string_numerics_df.columns}
        apply_casts(string_numerics_df, {"id": "Int64"})
        for col, dtype in original_dtypes.items():
            assert str(string_numerics_df[col].dtype) == dtype
