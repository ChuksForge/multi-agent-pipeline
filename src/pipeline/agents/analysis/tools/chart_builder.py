"""
agents/analysis/tools/chart_builder.py
────────────────────────────────────────
Vega-Lite v5 chart specification builder.

Emits JSON specs, never rendered images. The Report Agent embeds these
specs directly in Markdown output as fenced JSON blocks. Any Vega-Lite
compatible renderer (Altair, Observable, vega-embed) can render them.

Available builders:
  - build_timeseries_chart   — line chart with optional anomaly overlay
  - build_distribution_chart — histogram for a single numeric column
  - build_bar_chart          — categorical value counts
  - build_correlation_heatmap — numeric column correlation matrix
  - build_anomaly_scatter    — scatter plot coloured by anomaly flag

Each function validates the spec has required Vega-Lite top-level keys
before returning. Raises ValueError on invalid inputs (wrong column names,
empty data) so the Analysis Agent can degrade gracefully.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from pipeline.core.schemas import AnomalyRecord, ChartSpec
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

_VEGA_SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"
_REQUIRED_SPEC_KEYS = {"$schema", "data", "mark", "encoding"}


def build_timeseries_chart(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    anomalies: list[AnomalyRecord] | None = None,
    title: str | None = None,
) -> ChartSpec:
    """
    Line chart of y_col over x_col with optional anomaly point overlay.

    Args:
        df: Source DataFrame.
        x_col: Column to use as X axis (typically a date/timestamp column).
        y_col: Numeric column to plot on Y axis.
        anomalies: If provided, overlays red points at anomaly rows.
        title: Chart title. Defaults to "{y_col} over time".
    """
    _validate_columns(df, [x_col, y_col])

    records = df.select([x_col, y_col]).to_dicts()
    chart_title = title or f"{y_col} over {x_col}"

    # Base line layer
    base_layer: dict[str, Any] = {
        "mark": {"type": "line", "color": "#4C78A8", "strokeWidth": 2},
        "encoding": {
            "x": {"field": x_col, "type": _infer_vega_type(df[x_col].dtype), "title": x_col},
            "y": {"field": y_col, "type": "quantitative", "title": y_col},
        },
    }

    layers = [base_layer]

    # Anomaly overlay layer
    if anomalies:
        anomaly_rows = [
            {x_col: df[x_col][a.row_index], y_col: df[y_col][a.row_index]}
            for a in anomalies
            if a.row_index < len(df)
            and df[y_col][a.row_index] is not None
        ]
        if anomaly_rows:
            anomaly_layer: dict[str, Any] = {
                "data": {"values": anomaly_rows},
                "mark": {"type": "point", "color": "#E45756", "size": 80, "filled": True},
                "encoding": {
                    "x": {"field": x_col, "type": _infer_vega_type(df[x_col].dtype)},
                    "y": {"field": y_col, "type": "quantitative"},
                    "tooltip": [
                        {"field": x_col, "type": _infer_vega_type(df[x_col].dtype)},
                        {"field": y_col, "type": "quantitative"},
                    ],
                },
            }
            layers.append(anomaly_layer)

    spec: dict[str, Any] = {
        "$schema": _VEGA_SCHEMA,
        "title": chart_title,
        "data": {"values": records},
        "layer": layers,
        "width": 600,
        "height": 300,
    }

    # Layered charts use 'layer' not 'mark'/'encoding' at top level
    # Validate adjusted for layered spec
    _validate_spec_layered(spec)

    return ChartSpec(
        title=chart_title,
        spec=spec,
        description=f"Time series of {y_col} with {len(anomalies or [])} anomalies highlighted",
    )


def build_distribution_chart(
    df: pl.DataFrame,
    col: str,
    title: str | None = None,
    bin_count: int = 20,
) -> ChartSpec:
    """
    Histogram showing the distribution of a single numeric column.

    Args:
        df: Source DataFrame.
        col: Numeric column to histogram.
        title: Chart title. Defaults to "Distribution of {col}".
        bin_count: Number of histogram bins.
    """
    _validate_columns(df, [col])
    _validate_numeric(df, col)

    records = df.select(col).drop_nulls().to_dicts()
    chart_title = title or f"Distribution of {col}"

    spec: dict[str, Any] = {
        "$schema": _VEGA_SCHEMA,
        "title": chart_title,
        "data": {"values": records},
        "mark": {"type": "bar", "color": "#4C78A8"},
        "encoding": {
            "x": {
                "field": col,
                "type": "quantitative",
                "bin": {"maxbins": bin_count},
                "title": col,
            },
            "y": {
                "aggregate": "count",
                "type": "quantitative",
                "title": "Count",
            },
        },
        "width": 500,
        "height": 300,
    }

    _validate_spec(spec)
    return ChartSpec(
        title=chart_title,
        spec=spec,
        description=f"Histogram of {col} with {bin_count} bins",
    )


def build_bar_chart(
    df: pl.DataFrame,
    col: str,
    top_n: int = 15,
    title: str | None = None,
) -> ChartSpec:
    """
    Horizontal bar chart of value counts for a categorical column.

    Args:
        df: Source DataFrame.
        col: Categorical column.
        top_n: Show only the top N most frequent values.
        title: Chart title.
    """
    _validate_columns(df, [col])

    # Compute value counts
    vc = (
        df[col]
        .drop_nulls()
        .value_counts(sort=True)
        .head(top_n)
    )
    records = vc.to_dicts()
    count_col = "count"
    chart_title = title or f"Top {top_n} values in {col}"

    spec: dict[str, Any] = {
        "$schema": _VEGA_SCHEMA,
        "title": chart_title,
        "data": {"values": records},
        "mark": {"type": "bar", "color": "#72B7B2"},
        "encoding": {
            "y": {
                "field": col,
                "type": "nominal",
                "sort": "-x",
                "title": col,
            },
            "x": {
                "field": count_col,
                "type": "quantitative",
                "title": "Count",
            },
        },
        "width": 500,
        "height": max(200, top_n * 20),
    }

    _validate_spec(spec)
    return ChartSpec(
        title=chart_title,
        spec=spec,
        description=f"Top {top_n} most frequent values in {col}",
    )


def build_correlation_heatmap(
    df: pl.DataFrame,
    numeric_cols: list[str] | None = None,
    title: str | None = None,
) -> ChartSpec:
    """
    Heatmap of Pearson correlations between numeric columns.

    Args:
        df: Source DataFrame.
        numeric_cols: Subset of columns to include. Defaults to all numeric.
        title: Chart title.
    """
    from pipeline.agents.analysis.tools.stats_engine import numeric_column_names

    cols = numeric_cols or numeric_column_names(df)
    if len(cols) < 2:
        raise ValueError(
            f"Correlation heatmap requires at least 2 numeric columns, got {len(cols)}"
        )

    # Compute correlation matrix
    records = []
    for col_a in cols:
        for col_b in cols:
            try:
                corr = df.select(
                    pl.corr(col_a, col_b, method="pearson")
                ).item()
                corr_val = round(float(corr), 4) if corr is not None else 0.0
            except Exception:
                corr_val = 0.0
            records.append({"col_a": col_a, "col_b": col_b, "correlation": corr_val})

    chart_title = title or "Feature Correlation Heatmap"

    spec: dict[str, Any] = {
        "$schema": _VEGA_SCHEMA,
        "title": chart_title,
        "data": {"values": records},
        "mark": "rect",
        "encoding": {
            "x": {"field": "col_a", "type": "nominal", "title": ""},
            "y": {"field": "col_b", "type": "nominal", "title": ""},
            "color": {
                "field": "correlation",
                "type": "quantitative",
                "scale": {"scheme": "redblue", "domain": [-1, 1]},
                "title": "Pearson r",
            },
            "tooltip": [
                {"field": "col_a", "type": "nominal"},
                {"field": "col_b", "type": "nominal"},
                {"field": "correlation", "type": "quantitative", "format": ".3f"},
            ],
        },
        "width": {"step": 40},
        "height": {"step": 40},
    }

    _validate_spec(spec)
    return ChartSpec(
        title=chart_title,
        spec=spec,
        description=f"Pearson correlation heatmap for {len(cols)} numeric columns",
    )


def build_anomaly_scatter(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    anomalies: list[AnomalyRecord],
    title: str | None = None,
) -> ChartSpec:
    """
    Scatter plot coloured by anomaly flag (normal=blue, anomaly=red).

    Args:
        df: Source DataFrame.
        x_col: X-axis numeric column.
        y_col: Y-axis numeric column.
        anomalies: List of AnomalyRecord to flag in red.
        title: Chart title.
    """
    _validate_columns(df, [x_col, y_col])

    anomaly_indices = {a.row_index for a in anomalies}
    records = []
    for i in range(len(df)):
        x_val = df[x_col][i]
        y_val = df[y_col][i]
        if x_val is None or y_val is None:
            continue
        records.append({
            x_col: x_val,
            y_col: y_val,
            "anomaly": "Anomaly" if i in anomaly_indices else "Normal",
        })

    chart_title = title or f"{y_col} vs {x_col} (anomalies highlighted)"

    spec: dict[str, Any] = {
        "$schema": _VEGA_SCHEMA,
        "title": chart_title,
        "data": {"values": records},
        "mark": {"type": "point", "filled": True, "opacity": 0.7},
        "encoding": {
            "x": {"field": x_col, "type": "quantitative", "title": x_col},
            "y": {"field": y_col, "type": "quantitative", "title": y_col},
            "color": {
                "field": "anomaly",
                "type": "nominal",
                "scale": {
                    "domain": ["Normal", "Anomaly"],
                    "range": ["#4C78A8", "#E45756"],
                },
                "title": "Status",
            },
            "size": {
                "field": "anomaly",
                "type": "nominal",
                "scale": {"domain": ["Normal", "Anomaly"], "range": [30, 80]},
                "legend": None,
            },
            "tooltip": [
                {"field": x_col, "type": "quantitative"},
                {"field": y_col, "type": "quantitative"},
                {"field": "anomaly", "type": "nominal"},
            ],
        },
        "width": 500,
        "height": 350,
    }

    _validate_spec(spec)
    return ChartSpec(
        title=chart_title,
        spec=spec,
        description=f"Scatter plot of {x_col} vs {y_col} with anomaly colouring",
    )


# ── Auto chart selection ──────────────────────────────────────────────────────


def auto_select_charts(
    df: pl.DataFrame,
    anomalies: list[AnomalyRecord],
    max_charts: int = 4,
) -> list[ChartSpec]:
    """
    Automatically select and build the most informative charts for a DataFrame.

    Logic:
      1. If a time-like column exists + numeric column → timeseries
      2. First two numeric columns → scatter (with anomaly colouring)
      3. Most anomalous numeric column → distribution histogram
      4. If ≥2 numeric columns → correlation heatmap
    """
    from pipeline.agents.analysis.tools.stats_engine import numeric_column_names

    charts: list[ChartSpec] = []
    numeric_cols = numeric_column_names(df)
    time_cols = _detect_time_columns(df)

    # 1. Timeseries
    if time_cols and numeric_cols and len(charts) < max_charts:
        try:
            charts.append(build_timeseries_chart(
                df, x_col=time_cols[0], y_col=numeric_cols[0],
                anomalies=anomalies,
            ))
        except Exception as e:
            logger.warning("auto_chart_timeseries_failed", error=str(e))

    # 2. Anomaly scatter (only if we have ≥2 numeric cols)
    if len(numeric_cols) >= 2 and len(charts) < max_charts:
        try:
            charts.append(build_anomaly_scatter(
                df, x_col=numeric_cols[0], y_col=numeric_cols[1],
                anomalies=anomalies,
            ))
        except Exception as e:
            logger.warning("auto_chart_scatter_failed", error=str(e))

    # 3. Distribution of most anomalous column
    if numeric_cols and len(charts) < max_charts:
        anomaly_cols = [a.column for a in anomalies if a.column in numeric_cols]
        dist_col = anomaly_cols[0] if anomaly_cols else numeric_cols[0]
        try:
            charts.append(build_distribution_chart(df, col=dist_col))
        except Exception as e:
            logger.warning("auto_chart_distribution_failed", error=str(e))

    # 4. Correlation heatmap
    if len(numeric_cols) >= 2 and len(charts) < max_charts:
        try:
            charts.append(build_correlation_heatmap(df, numeric_cols=numeric_cols[:6]))
        except Exception as e:
            logger.warning("auto_chart_heatmap_failed", error=str(e))

    logger.debug("auto_charts_built", count=len(charts))
    return charts


# ── Validation helpers ────────────────────────────────────────────────────────


def _validate_columns(df: pl.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")


def _validate_numeric(df: pl.DataFrame, col: str) -> None:
    from pipeline.agents.analysis.tools.stats_engine import _NUMERIC_DTYPES
    if df[col].dtype not in _NUMERIC_DTYPES:
        raise ValueError(f"Column '{col}' is not numeric (dtype={df[col].dtype})")


def _validate_spec(spec: dict[str, Any]) -> None:
    """Ensure a standard (non-layered) Vega-Lite spec has all required keys."""
    missing = _REQUIRED_SPEC_KEYS - set(spec.keys())
    if missing:
        raise ValueError(f"Vega-Lite spec missing required keys: {missing}")


def _validate_spec_layered(spec: dict[str, Any]) -> None:
    """Validate a layered Vega-Lite spec (uses 'layer' instead of 'mark'/'encoding')."""
    required = {"$schema", "data", "layer"}
    missing = required - set(spec.keys())
    if missing:
        raise ValueError(f"Layered Vega-Lite spec missing required keys: {missing}")


def _infer_vega_type(dtype: pl.PolarsDataType) -> str:
    """Map polars dtype to Vega-Lite encoding type string."""
    if dtype in (pl.Date, pl.Datetime, pl.Time):
        return "temporal"
    if dtype == pl.Utf8:
        return "nominal"
    if dtype == pl.Boolean:
        return "nominal"
    return "quantitative"


def _detect_time_columns(df: pl.DataFrame) -> list[str]:
    """
    Heuristic: find columns that look like timestamps.
    Checks dtype first, then column name patterns.
    """
    time_cols = []
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in (pl.Date, pl.Datetime):
            time_cols.append(col)
        elif dtype == pl.Utf8:
            name_lower = col.lower()
            if any(kw in name_lower for kw in ("date", "time", "ts", "timestamp", "created", "updated")):
                time_cols.append(col)
    return time_cols
