"""
agents/analysis/tools/anomaly_detector.py
──────────────────────────────────────────
Anomaly detection using IsolationForest + Z-score ensemble.

Strategy:
  - IsolationForest: fits on all numeric columns jointly, scores every row.
    Catches multivariate anomalies (e.g. unusual combinations of values).
  - Z-score: per-column, flags rows where |z| > threshold (default 3σ).
    Catches univariate outliers in individual columns.
  - Ensemble: a row is flagged if EITHER method flags it.
    AnomalyRecord.method set to "ensemble", "isolation_forest", or "zscore".

OOM / large data recovery:
  - If input > max_rows_before_sample, subsample before fitting IsolationForest.
  - Z-score always runs on the full dataset (it's O(n), no fitting cost).
  - RecoveryTier.SIMPLIFIED returned when sampling was applied.

Returns:
  - list[AnomalyRecord] — one record per flagged row
  - RecoveryTier — NONE if full data used, SIMPLIFIED if sampled
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest

from pipeline.agents.analysis.tools.stats_engine import numeric_column_names
from pipeline.core.exceptions import AnomalyDetectionError, InsufficientDataError
from pipeline.core.schemas import AnomalyRecord, RecoveryTier
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

_MIN_ROWS_FOR_ISOLATION_FOREST = 10
_DEFAULT_MAX_ROWS = 100_000
_DEFAULT_CONTAMINATION = "auto"
_DEFAULT_ZSCORE_THRESHOLD = 3.0
_DEFAULT_SAMPLE_RATIO = 0.10


@dataclass
class AnomalyConfig:
    """Configuration for the anomaly detector."""
    contamination: float | str = _DEFAULT_CONTAMINATION
    zscore_threshold: float = _DEFAULT_ZSCORE_THRESHOLD
    max_rows_before_sample: int = _DEFAULT_MAX_ROWS
    sample_ratio: float = _DEFAULT_SAMPLE_RATIO
    random_state: int = 42
    # Which methods to run
    run_isolation_forest: bool = True
    run_zscore: bool = True


def detect_anomalies(
    df: pl.DataFrame,
    config: AnomalyConfig | None = None,
) -> tuple[list[AnomalyRecord], RecoveryTier]:
    """
    Run anomaly detection on df. Returns (anomalies, recovery_tier).

    Args:
        df: Input DataFrame. Must have at least one numeric column.
        config: Detection configuration. Uses defaults if None.

    Returns:
        anomalies: List of AnomalyRecord for every flagged row.
        recovery_tier: NONE if full data used, SIMPLIFIED if OOM-sampled.

    Raises:
        InsufficientDataError: Fewer than _MIN_ROWS_FOR_ISOLATION_FOREST rows.
        AnomalyDetectionError: Both methods failed catastrophically.
    """
    cfg = config or AnomalyConfig()
    n_rows = len(df)
    recovery_tier = RecoveryTier.NONE

    numeric_cols = numeric_column_names(df)
    if not numeric_cols:
        logger.warning("anomaly_detection_no_numeric_cols")
        return [], RecoveryTier.NONE

    if n_rows < _MIN_ROWS_FOR_ISOLATION_FOREST:
        raise InsufficientDataError(
            required=_MIN_ROWS_FOR_ISOLATION_FOREST,
            actual=n_rows,
        )

    # Drop rows where ALL numeric cols are null — can't score these
    numeric_df = df.select(numeric_cols).drop_nulls()
    valid_indices = _get_valid_indices(df, numeric_cols)

    if len(numeric_df) < _MIN_ROWS_FOR_ISOLATION_FOREST:
        raise InsufficientDataError(
            required=_MIN_ROWS_FOR_ISOLATION_FOREST,
            actual=len(numeric_df),
        )

    # ── IsolationForest ───────────────────────────────────────────────────────
    if_anomaly_indices: set[int] = set()
    if_scores: dict[int, float] = {}

    if cfg.run_isolation_forest:
        fit_df = numeric_df
        fit_indices = valid_indices

        # OOM guard: sample if too many rows
        if len(numeric_df) > cfg.max_rows_before_sample:
            sample_n = max(
                _MIN_ROWS_FOR_ISOLATION_FOREST,
                int(len(numeric_df) * cfg.sample_ratio),
            )
            logger.warning(
                "isolation_forest_sampling",
                original_rows=len(numeric_df),
                sample_rows=sample_n,
            )
            sample_mask = np.random.default_rng(cfg.random_state).choice(
                len(numeric_df), size=sample_n, replace=False
            )
            fit_df = numeric_df[sample_mask.tolist()]
            fit_indices = [valid_indices[i] for i in sample_mask]
            recovery_tier = RecoveryTier.SIMPLIFIED

        try:
            X_fit = fit_df.fill_null(0).to_numpy()
            X_score = numeric_df.fill_null(0).to_numpy()

            iso = IsolationForest(
                contamination=cfg.contamination,
                random_state=cfg.random_state,
                n_jobs=-1,
            )
            iso.fit(X_fit)
            predictions = iso.predict(X_score)   # -1 = anomaly, 1 = normal
            scores = iso.score_samples(X_score)  # lower = more anomalous

            for i, (pred, score) in enumerate(zip(predictions, scores)):
                orig_idx = valid_indices[i]
                if_scores[orig_idx] = float(score)
                if pred == -1:
                    if_anomaly_indices.add(orig_idx)

            logger.debug(
                "isolation_forest_complete",
                flagged=len(if_anomaly_indices),
                total_scored=len(numeric_df),
            )
        except MemoryError:
            logger.warning("isolation_forest_oom", note="Falling back to Z-score only")
            recovery_tier = RecoveryTier.SIMPLIFIED
        except Exception as e:
            logger.warning("isolation_forest_failed", error=str(e))

    # ── Z-score (always runs on full data) ────────────────────────────────────
    zscore_anomaly_indices: set[int] = set()
    zscore_col_map: dict[int, str] = {}  # row_idx → which column triggered it

    if cfg.run_zscore:
        for col_name in numeric_cols:
            series = df[col_name].drop_nulls()
            if len(series) < 3:
                continue
            try:
                arr = series.cast(pl.Float64).to_numpy()
                z_scores = np.abs(scipy_stats.zscore(arr, nan_policy="omit"))

                # Map back to original DataFrame indices
                non_null_indices = [
                    i for i in range(n_rows)
                    if df[col_name][i] is not None
                ]
                for local_i, (orig_idx, z) in enumerate(zip(non_null_indices, z_scores)):
                    if np.isnan(z):
                        continue
                    if z > cfg.zscore_threshold:
                        zscore_anomaly_indices.add(orig_idx)
                        # Only store the column if not already flagged
                        if orig_idx not in zscore_col_map:
                            zscore_col_map[orig_idx] = col_name
            except Exception as e:
                logger.warning("zscore_failed", column=col_name, error=str(e))

        logger.debug(
            "zscore_complete",
            flagged=len(zscore_anomaly_indices),
        )

    # ── Build ensemble AnomalyRecord list ────────────────────────────────────
    if not cfg.run_isolation_forest and not cfg.run_zscore:
        raise AnomalyDetectionError("Both detection methods disabled in config")

    all_flagged = if_anomaly_indices | zscore_anomaly_indices
    if not all_flagged and not if_anomaly_indices and not zscore_anomaly_indices:
        logger.info("anomaly_detection_complete", anomalies=0)
        return [], recovery_tier

    records: list[AnomalyRecord] = []
    for row_idx in sorted(all_flagged):
        in_if = row_idx in if_anomaly_indices
        in_z = row_idx in zscore_anomaly_indices

        if in_if and in_z:
            method = "ensemble"
        elif in_if:
            method = "isolation_forest"
        else:
            method = "zscore"

        # Score: use IF score if available, else use a synthetic z-based score
        score = if_scores.get(row_idx, -1.0)

        # Best column to report: use zscore column if available, else first numeric
        col = zscore_col_map.get(row_idx, numeric_cols[0])

        try:
            value = df[col][row_idx]
        except Exception:
            value = None

        records.append(AnomalyRecord(
            row_index=row_idx,
            column=col,
            value=value,
            anomaly_score=round(score, 4),
            method=method,
            is_anomaly=True,
        ))

    logger.info(
        "anomaly_detection_complete",
        total_rows=n_rows,
        anomalies=len(records),
        anomaly_rate=round(len(records) / n_rows, 4),
        recovery_tier=recovery_tier.value,
    )
    return records, recovery_tier


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_valid_indices(df: pl.DataFrame, numeric_cols: list[str]) -> list[int]:
    """
    Return row indices where at least one numeric column is non-null.
    These are the rows IsolationForest can score.
    """
    mask = pl.Series([False] * len(df))
    for col in numeric_cols:
        mask = mask | df[col].is_not_null()
    return [i for i, v in enumerate(mask.to_list()) if v]


def anomaly_rate(records: list[AnomalyRecord], total_rows: int) -> float:
    """Compute fraction of rows flagged as anomalies."""
    if total_rows == 0:
        return 0.0
    return round(len(records) / total_rows, 4)
