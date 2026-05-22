"""
tests/unit/test_stats_engine.py
─────────────────────────────────
Unit tests for stats_engine: compute_stats, compute_numeric_stats_only,
numeric_column_names. Uses real polars DataFrames — no mocking.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.analysis.tools.stats_engine import (
    compute_stats,
    compute_numeric_stats_only,
    numeric_column_names,
)
from pipeline.core.schemas import SummaryStats


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def numeric_df():
    return pl.DataFrame({
        "revenue": [100.0, 200.0, 150.0, 300.0, 250.0, 175.0, 225.0, 125.0],
        "units": [10, 20, 15, 30, 25, 17, 22, 12],
        "cost": [40.0, 80.0, 60.0, 120.0, 100.0, 70.0, 90.0, 50.0],
    })


@pytest.fixture
def mixed_df():
    return pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        "revenue": [1000.0, 1200.0, 950.0, 1500.0],
        "region": ["north", "south", "east", "north"],
        "active": [True, False, True, True],
    })


@pytest.fixture
def nulls_df():
    return pl.DataFrame({
        "a": [1.0, None, 3.0, None, 5.0],
        "b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def all_null_df():
    return pl.DataFrame({
        "x": [None, None, None],
        "y": [1.0, 2.0, 3.0],
    })


@pytest.fixture
def categorical_df():
    return pl.DataFrame({
        "category": ["a", "b", "a", "c", "a", "b"],
        "count": [10, 20, 30, 40, 50, 60],
    })


# ─── compute_stats ────────────────────────────────────────────────────────────


class TestComputeStats:
    def test_returns_one_record_per_column(self, numeric_df):
        stats = compute_stats(numeric_df)
        assert len(stats) == 3

    def test_all_are_summary_stats(self, numeric_df):
        stats = compute_stats(numeric_df)
        assert all(isinstance(s, SummaryStats) for s in stats)

    def test_column_names_preserved(self, numeric_df):
        stats = compute_stats(numeric_df)
        names = {s.column for s in stats}
        assert names == {"revenue", "units", "cost"}

    def test_numeric_mean_correct(self, numeric_df):
        stats = compute_stats(numeric_df)
        revenue = next(s for s in stats if s.column == "revenue")
        expected_mean = sum([100, 200, 150, 300, 250, 175, 225, 125]) / 8
        assert revenue.mean == pytest.approx(expected_mean)

    def test_numeric_min_max_correct(self, numeric_df):
        stats = compute_stats(numeric_df)
        revenue = next(s for s in stats if s.column == "revenue")
        assert revenue.min == pytest.approx(100.0)
        assert revenue.max == pytest.approx(300.0)

    def test_null_count_zero_for_clean_data(self, numeric_df):
        stats = compute_stats(numeric_df)
        for s in stats:
            assert s.null_count == 0

    def test_null_count_correct_with_nulls(self, nulls_df):
        stats = compute_stats(nulls_df)
        a_stats = next(s for s in stats if s.column == "a")
        assert a_stats.null_count == 2

    def test_count_excludes_nulls(self, nulls_df):
        stats = compute_stats(nulls_df)
        a_stats = next(s for s in stats if s.column == "a")
        assert a_stats.count == 3  # 5 rows - 2 nulls

    def test_percentiles_populated(self, numeric_df):
        stats = compute_stats(numeric_df)
        revenue = next(s for s in stats if s.column == "revenue")
        assert revenue.p25 is not None
        assert revenue.p50 is not None
        assert revenue.p75 is not None
        assert revenue.p25 <= revenue.p50 <= revenue.p75

    def test_std_positive_for_varied_data(self, numeric_df):
        stats = compute_stats(numeric_df)
        revenue = next(s for s in stats if s.column == "revenue")
        assert revenue.std is not None
        assert revenue.std > 0

    def test_std_zero_for_constant_column(self):
        df = pl.DataFrame({"x": [5.0, 5.0, 5.0, 5.0]})
        stats = compute_stats(df)
        assert stats[0].std == pytest.approx(0.0)

    def test_categorical_column_gets_top_values(self, categorical_df):
        stats = compute_stats(categorical_df)
        cat_stats = next(s for s in stats if s.column == "category")
        assert len(cat_stats.top_values) > 0
        assert "a" in cat_stats.top_values  # "a" appears 3 times — most frequent

    def test_categorical_no_mean_or_std(self, categorical_df):
        stats = compute_stats(categorical_df)
        cat_stats = next(s for s in stats if s.column == "category")
        assert cat_stats.mean is None
        assert cat_stats.std is None

    def test_mixed_df_handles_all_types(self, mixed_df):
        stats = compute_stats(mixed_df)
        assert len(stats) == 4
        names = {s.column for s in stats}
        assert names == {"date", "revenue", "region", "active"}

    def test_empty_dataframe_returns_empty_list(self):
        df = pl.DataFrame()
        stats = compute_stats(df)
        assert stats == []

    def test_entirely_null_numeric_column(self, all_null_df):
        stats = compute_stats(all_null_df)
        x_stats = next(s for s in stats if s.column == "x")
        assert x_stats.count == 0
        assert x_stats.null_count == 3
        assert x_stats.mean is None

    def test_dtype_string_populated(self, numeric_df):
        stats = compute_stats(numeric_df)
        for s in stats:
            assert isinstance(s.dtype, str)
            assert len(s.dtype) > 0


# ─── compute_numeric_stats_only ───────────────────────────────────────────────


class TestComputeNumericStatsOnly:
    def test_only_numeric_columns_included(self, mixed_df):
        stats = compute_numeric_stats_only(mixed_df)
        col_names = {s.column for s in stats}
        assert "revenue" in col_names
        assert "region" not in col_names
        assert "date" not in col_names

    def test_returns_empty_for_no_numeric_cols(self):
        df = pl.DataFrame({"name": ["a", "b"], "label": ["x", "y"]})
        stats = compute_numeric_stats_only(df)
        assert stats == []


# ─── numeric_column_names ─────────────────────────────────────────────────────


class TestNumericColumnNames:
    def test_returns_numeric_columns(self, mixed_df):
        cols = numeric_column_names(mixed_df)
        assert "revenue" in cols
        assert "region" not in cols

    def test_returns_empty_for_no_numeric(self):
        df = pl.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
        assert numeric_column_names(df) == []

    def test_all_int_and_float_types_detected(self):
        df = pl.DataFrame({
            "i8": pl.Series([1, 2], dtype=pl.Int8),
            "i64": pl.Series([1, 2], dtype=pl.Int64),
            "f32": pl.Series([1.0, 2.0], dtype=pl.Float32),
            "f64": pl.Series([1.0, 2.0], dtype=pl.Float64),
            "str": ["a", "b"],
        })
        cols = numeric_column_names(df)
        assert set(cols) == {"i8", "i64", "f32", "f64"}
