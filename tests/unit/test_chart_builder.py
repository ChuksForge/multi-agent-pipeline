"""
tests/unit/test_chart_builder.py
──────────────────────────────────
Unit tests for chart_builder. Validates Vega-Lite spec structure —
never renders images. Tests that required keys are present and
data is correctly embedded in specs.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl

from pipeline.agents.analysis.tools.chart_builder import (
    auto_select_charts,
    build_anomaly_scatter,
    build_bar_chart,
    build_correlation_heatmap,
    build_distribution_chart,
    build_timeseries_chart,
)
from pipeline.core.schemas import AnomalyRecord, ChartSpec

_VEGA_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def timeseries_df():
    return pl.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "revenue": [100.0, 120.0, 95.0, 500.0, 110.0],
    })


@pytest.fixture
def numeric_df():
    return pl.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [10.0, 20.0, 15.0, 100.0, 25.0, 18.0, 22.0, 12.0],
        "z": [5.0, 4.0, 6.0, 3.0, 7.0, 4.5, 5.5, 6.5],
    })


@pytest.fixture
def categorical_df():
    return pl.DataFrame({
        "region": ["north", "south", "east", "north", "west", "south", "north"],
        "value": [100.0, 200.0, 150.0, 120.0, 180.0, 210.0, 90.0],
    })


@pytest.fixture
def sample_anomalies():
    return [
        AnomalyRecord(row_index=3, column="revenue", value=500.0,
                      anomaly_score=-0.45, method="ensemble"),
    ]


# ─── Spec structure validation ────────────────────────────────────────────────


def _assert_valid_spec(chart: ChartSpec) -> None:
    """Assert a ChartSpec has all required Vega-Lite fields."""
    assert isinstance(chart, ChartSpec)
    assert "$schema" in chart.spec
    assert chart.spec["$schema"] == _VEGA_SCHEMA
    assert "data" in chart.spec
    assert "title" in chart.spec or chart.title
    assert isinstance(chart.title, str) and len(chart.title) > 0
    assert isinstance(chart.description, str)


# ─── build_timeseries_chart ───────────────────────────────────────────────────


class TestBuildTimeseriesChart:
    def test_returns_chart_spec(self, timeseries_df):
        chart = build_timeseries_chart(timeseries_df, x_col="date", y_col="revenue")
        assert isinstance(chart, ChartSpec)

    def test_vega_schema_present(self, timeseries_df):
        chart = build_timeseries_chart(timeseries_df, x_col="date", y_col="revenue")
        assert chart.spec["$schema"] == _VEGA_SCHEMA

    def test_data_embedded_in_spec(self, timeseries_df):
        chart = build_timeseries_chart(timeseries_df, x_col="date", y_col="revenue")
        assert "data" in chart.spec
        assert "values" in chart.spec["data"]
        assert len(chart.spec["data"]["values"]) == 5

    def test_layer_key_present(self, timeseries_df):
        """Timeseries uses layered spec (base line + optional anomaly overlay)."""
        chart = build_timeseries_chart(timeseries_df, x_col="date", y_col="revenue")
        assert "layer" in chart.spec

    def test_anomaly_overlay_added_when_provided(self, timeseries_df, sample_anomalies):
        chart = build_timeseries_chart(
            timeseries_df, x_col="date", y_col="revenue",
            anomalies=sample_anomalies,
        )
        # Should have 2 layers: base line + anomaly points
        assert len(chart.spec["layer"]) == 2

    def test_no_overlay_without_anomalies(self, timeseries_df):
        chart = build_timeseries_chart(timeseries_df, x_col="date", y_col="revenue")
        assert len(chart.spec["layer"]) == 1

    def test_custom_title_used(self, timeseries_df):
        chart = build_timeseries_chart(
            timeseries_df, x_col="date", y_col="revenue", title="My Chart"
        )
        assert chart.title == "My Chart"

    def test_missing_column_raises(self, timeseries_df):
        with pytest.raises(ValueError, match="not found"):
            build_timeseries_chart(timeseries_df, x_col="date", y_col="nonexistent")


# ─── build_distribution_chart ─────────────────────────────────────────────────


class TestBuildDistributionChart:
    def test_returns_chart_spec(self, numeric_df):
        chart = build_distribution_chart(numeric_df, col="x")
        _assert_valid_spec(chart)

    def test_mark_is_bar(self, numeric_df):
        chart = build_distribution_chart(numeric_df, col="x")
        mark = chart.spec["mark"]
        mark_type = mark if isinstance(mark, str) else mark.get("type")
        assert mark_type == "bar"

    def test_encoding_has_x_and_y(self, numeric_df):
        chart = build_distribution_chart(numeric_df, col="x")
        enc = chart.spec["encoding"]
        assert "x" in enc
        assert "y" in enc

    def test_x_field_is_target_column(self, numeric_df):
        chart = build_distribution_chart(numeric_df, col="x")
        assert chart.spec["encoding"]["x"]["field"] == "x"

    def test_non_numeric_column_raises(self, categorical_df):
        with pytest.raises(ValueError):
            build_distribution_chart(categorical_df, col="region")

    def test_missing_column_raises(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            build_distribution_chart(numeric_df, col="nonexistent")


# ─── build_bar_chart ──────────────────────────────────────────────────────────


class TestBuildBarChart:
    def test_returns_chart_spec(self, categorical_df):
        chart = build_bar_chart(categorical_df, col="region")
        _assert_valid_spec(chart)

    def test_mark_is_bar(self, categorical_df):
        chart = build_bar_chart(categorical_df, col="region")
        mark = chart.spec["mark"]
        mark_type = mark if isinstance(mark, str) else mark.get("type")
        assert mark_type == "bar"

    def test_data_values_present(self, categorical_df):
        chart = build_bar_chart(categorical_df, col="region")
        assert len(chart.spec["data"]["values"]) > 0

    def test_top_n_limit_respected(self, categorical_df):
        chart = build_bar_chart(categorical_df, col="region", top_n=2)
        assert len(chart.spec["data"]["values"]) <= 2

    def test_missing_column_raises(self, categorical_df):
        with pytest.raises(ValueError, match="not found"):
            build_bar_chart(categorical_df, col="nonexistent")


# ─── build_correlation_heatmap ────────────────────────────────────────────────


class TestBuildCorrelationHeatmap:
    def test_returns_chart_spec(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df)
        _assert_valid_spec(chart)

    def test_mark_is_rect(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df)
        mark = chart.spec["mark"]
        mark_type = mark if isinstance(mark, str) else mark.get("type", mark)
        assert mark_type == "rect"

    def test_data_has_correlation_values(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df)
        values = chart.spec["data"]["values"]
        assert len(values) == 9  # 3x3 matrix
        # All records have col_a, col_b, correlation
        for v in values:
            assert "col_a" in v
            assert "col_b" in v
            assert "correlation" in v

    def test_diagonal_is_one(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df)
        values = chart.spec["data"]["values"]
        diagonal = [v for v in values if v["col_a"] == v["col_b"]]
        for d in diagonal:
            assert d["correlation"] == pytest.approx(1.0, abs=1e-4)

    def test_correlations_bounded(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df)
        for v in chart.spec["data"]["values"]:
            assert -1.0 <= v["correlation"] <= 1.0

    def test_fewer_than_two_columns_raises(self):
        df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="at least 2"):
            build_correlation_heatmap(df)

    def test_subset_of_columns(self, numeric_df):
        chart = build_correlation_heatmap(numeric_df, numeric_cols=["x", "y"])
        values = chart.spec["data"]["values"]
        assert len(values) == 4  # 2x2 matrix


# ─── build_anomaly_scatter ────────────────────────────────────────────────────


class TestBuildAnomalyScatter:
    def test_returns_chart_spec(self, numeric_df, sample_anomalies):
        chart = build_anomaly_scatter(
            numeric_df, x_col="x", y_col="y", anomalies=sample_anomalies
        )
        _assert_valid_spec(chart)

    def test_anomaly_field_present_in_data(self, numeric_df, sample_anomalies):
        chart = build_anomaly_scatter(
            numeric_df, x_col="x", y_col="y", anomalies=sample_anomalies
        )
        values = chart.spec["data"]["values"]
        anomaly_values = {v["anomaly"] for v in values}
        assert "Anomaly" in anomaly_values
        assert "Normal" in anomaly_values

    def test_color_encoding_present(self, numeric_df, sample_anomalies):
        chart = build_anomaly_scatter(
            numeric_df, x_col="x", y_col="y", anomalies=sample_anomalies
        )
        assert "color" in chart.spec["encoding"]

    def test_empty_anomalies_still_works(self, numeric_df):
        chart = build_anomaly_scatter(numeric_df, x_col="x", y_col="y", anomalies=[])
        _assert_valid_spec(chart)
        values = chart.spec["data"]["values"]
        assert all(v["anomaly"] == "Normal" for v in values)

    def test_missing_columns_raise(self, numeric_df):
        with pytest.raises(ValueError, match="not found"):
            build_anomaly_scatter(numeric_df, x_col="x", y_col="bad_col", anomalies=[])


# ─── auto_select_charts ───────────────────────────────────────────────────────


class TestAutoSelectCharts:
    def test_returns_list_of_chart_specs(self, timeseries_df, sample_anomalies):
        charts = auto_select_charts(timeseries_df, anomalies=sample_anomalies)
        assert isinstance(charts, list)
        assert all(isinstance(c, ChartSpec) for c in charts)

    def test_respects_max_charts(self, numeric_df):
        charts = auto_select_charts(numeric_df, anomalies=[], max_charts=2)
        assert len(charts) <= 2

    def test_returns_at_least_one_chart_for_numeric_data(self, numeric_df):
        charts = auto_select_charts(numeric_df, anomalies=[])
        assert len(charts) >= 1

    def test_timeseries_chart_generated_for_time_columns(self, timeseries_df):
        charts = auto_select_charts(timeseries_df, anomalies=[])
        chart_titles = [c.title.lower() for c in charts]
        # Should have generated a timeseries chart
        assert any("revenue" in t or "date" in t for t in chart_titles)

    def test_no_crash_on_empty_anomalies(self, numeric_df):
        charts = auto_select_charts(numeric_df, anomalies=[])
        assert isinstance(charts, list)

    def test_all_specs_have_vega_schema(self, numeric_df, sample_anomalies):
        charts = auto_select_charts(numeric_df, anomalies=sample_anomalies)
        for chart in charts:
            assert chart.spec["$schema"] == _VEGA_SCHEMA
