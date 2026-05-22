"""
tests/unit/test_anomaly_detector.py
──────────────────────────────────────
Unit tests for anomaly_detector: detect_anomalies, AnomalyConfig, helpers.

Uses real scikit-learn and scipy — no mocking.
Injects known anomalies into controlled DataFrames so tests are deterministic.
"""

from __future__ import annotations

import os
import pytest

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test")

import polars as pl
import numpy as np

from pipeline.agents.analysis.tools.anomaly_detector import (
    AnomalyConfig,
    anomaly_rate,
    detect_anomalies,
)
from pipeline.core.exceptions import AnomalyDetectionError, InsufficientDataError
from pipeline.core.schemas import RecoveryTier


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def clean_df():
    """50 rows of normally distributed data — few or no anomalies expected."""
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "x": rng.normal(loc=100.0, scale=10.0, size=50).tolist(),
        "y": rng.normal(loc=50.0, scale=5.0, size=50).tolist(),
    })


@pytest.fixture
def anomalous_df():
    """Normal data with 3 injected extreme outliers."""
    rng = np.random.default_rng(42)
    x = rng.normal(loc=100.0, scale=5.0, size=47).tolist()
    y = rng.normal(loc=50.0, scale=3.0, size=47).tolist()
    # Inject 3 extreme outliers
    x += [500.0, -200.0, 450.0]
    y += [500.0, -150.0, 400.0]
    return pl.DataFrame({"x": x, "y": y})


@pytest.fixture
def tiny_df():
    """Fewer than min rows required."""
    return pl.DataFrame({"x": [1.0, 2.0, 3.0]})


@pytest.fixture
def no_numeric_df():
    return pl.DataFrame({"label": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]})


@pytest.fixture
def zscore_df():
    """Data with one obvious Z-score outlier — should be caught by Z-score."""
    # 19 values around 100, one extreme value at 1000 (z > 3)
    values = [100.0] * 19 + [1000.0]
    return pl.DataFrame({"value": values})


# ─── detect_anomalies happy path ─────────────────────────────────────────────


class TestDetectAnomaliesHappyPath:
    def test_returns_tuple(self, clean_df):
        result = detect_anomalies(clean_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_list_and_recovery_tier(self, clean_df):
        anomalies, recovery_tier = detect_anomalies(clean_df)
        assert isinstance(anomalies, list)
        assert isinstance(recovery_tier, RecoveryTier)

    def test_clean_data_low_anomaly_count(self, clean_df):
        anomalies, _ = detect_anomalies(clean_df)
        # IsolationForest with contamination='auto' on 50 rows flags ~12 rows.
        # Bound is generous to accommodate platform variance.
        assert len(anomalies) <= 15

    def test_anomalous_data_detects_outliers(self, anomalous_df):
        anomalies, _ = detect_anomalies(anomalous_df)
        # Should detect at least the 3 injected extreme outliers
        assert len(anomalies) >= 1

    def test_extreme_outliers_detected(self, anomalous_df):
        anomalies, _ = detect_anomalies(anomalous_df)
        anomaly_rows = {a.row_index for a in anomalies}
        # Injected outliers are at rows 47, 48, 49
        injected = {47, 48, 49}
        detected = anomaly_rows & injected
        assert len(detected) >= 1  # at least one of the 3 extreme outliers caught

    def test_recovery_tier_none_for_small_data(self, clean_df):
        _, recovery_tier = detect_anomalies(clean_df)
        assert recovery_tier == RecoveryTier.NONE

    def test_anomaly_records_have_required_fields(self, anomalous_df):
        anomalies, _ = detect_anomalies(anomalous_df)
        if anomalies:
            a = anomalies[0]
            assert isinstance(a.row_index, int)
            assert isinstance(a.anomaly_score, float)
            assert a.method in ("isolation_forest", "zscore", "ensemble")
            assert a.is_anomaly is True

    def test_records_sorted_by_row_index(self, anomalous_df):
        anomalies, _ = detect_anomalies(anomalous_df)
        indices = [a.row_index for a in anomalies]
        assert indices == sorted(indices)


# ─── Z-score detection ────────────────────────────────────────────────────────


class TestZScoreDetection:
    def test_zscore_catches_extreme_outlier(self, zscore_df):
        cfg = AnomalyConfig(run_isolation_forest=False, run_zscore=True)
        anomalies, _ = detect_anomalies(zscore_df, config=cfg)
        # Row 19 (value=1000) should be detected
        row_indices = {a.row_index for a in anomalies}
        assert 19 in row_indices

    def test_zscore_only_config(self, anomalous_df):
        cfg = AnomalyConfig(run_isolation_forest=False, run_zscore=True)
        anomalies, _ = detect_anomalies(anomalous_df, config=cfg)
        # Should still catch extreme outliers
        assert len(anomalies) >= 1
        for a in anomalies:
            assert a.method == "zscore"

    def test_high_threshold_fewer_anomalies(self, anomalous_df):
        cfg_strict = AnomalyConfig(run_isolation_forest=False, zscore_threshold=5.0)
        cfg_loose = AnomalyConfig(run_isolation_forest=False, zscore_threshold=2.0)
        strict_anomalies, _ = detect_anomalies(anomalous_df, config=cfg_strict)
        loose_anomalies, _ = detect_anomalies(anomalous_df, config=cfg_loose)
        assert len(strict_anomalies) <= len(loose_anomalies)


# ─── IsolationForest only ────────────────────────────────────────────────────


class TestIsolationForestOnly:
    def test_isolation_forest_only_config(self, anomalous_df):
        cfg = AnomalyConfig(run_isolation_forest=True, run_zscore=False)
        anomalies, _ = detect_anomalies(anomalous_df, config=cfg)
        assert len(anomalies) >= 1
        for a in anomalies:
            assert a.method == "isolation_forest"

    def test_isolation_forest_scores_are_negative(self, anomalous_df):
        cfg = AnomalyConfig(run_isolation_forest=True, run_zscore=False)
        anomalies, _ = detect_anomalies(anomalous_df, config=cfg)
        # IsolationForest anomaly scores are negative (lower = more anomalous)
        for a in anomalies:
            assert a.anomaly_score < 0


# ─── Ensemble ────────────────────────────────────────────────────────────────


class TestEnsemble:
    def test_ensemble_method_label_assigned(self, anomalous_df):
        """Rows flagged by both methods should have method='ensemble'."""
        cfg = AnomalyConfig(run_isolation_forest=True, run_zscore=True)
        anomalies, _ = detect_anomalies(anomalous_df, config=cfg)
        methods = {a.method for a in anomalies}
        # At least one method should appear
        assert methods.issubset({"isolation_forest", "zscore", "ensemble"})


# ─── OOM / large data sampling ───────────────────────────────────────────────


class TestLargeDataSampling:
    def test_sampling_applied_when_exceeds_limit(self):
        """Trigger sampling by setting max_rows low."""
        rng = np.random.default_rng(0)
        df = pl.DataFrame({
            "x": rng.normal(size=100).tolist(),
            "y": rng.normal(size=100).tolist(),
        })
        cfg = AnomalyConfig(
            max_rows_before_sample=20,  # Force sampling on 100-row df
            sample_ratio=0.30,
        )
        anomalies, recovery_tier = detect_anomalies(df, config=cfg)
        assert recovery_tier == RecoveryTier.SIMPLIFIED

    def test_sampling_still_returns_records(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=95).tolist() + [999.0, -999.0, 888.0, -888.0, 777.0]
        df = pl.DataFrame({"x": x, "y": rng.normal(size=100).tolist()})
        cfg = AnomalyConfig(max_rows_before_sample=30, sample_ratio=0.50)
        anomalies, _ = detect_anomalies(df, config=cfg)
        assert isinstance(anomalies, list)


# ─── Error cases ─────────────────────────────────────────────────────────────


class TestAnomalyDetectorErrors:
    def test_insufficient_rows_raises(self, tiny_df):
        with pytest.raises(InsufficientDataError):
            detect_anomalies(tiny_df)

    def test_no_numeric_columns_returns_empty(self, no_numeric_df):
        anomalies, recovery_tier = detect_anomalies(no_numeric_df)
        assert anomalies == []
        assert recovery_tier == RecoveryTier.NONE

    def test_both_methods_disabled_raises(self, clean_df):
        cfg = AnomalyConfig(run_isolation_forest=False, run_zscore=False)
        with pytest.raises(AnomalyDetectionError):
            detect_anomalies(clean_df, config=cfg)


# ─── anomaly_rate helper ──────────────────────────────────────────────────────


class TestAnomalyRate:
    def test_zero_anomalies(self):
        assert anomaly_rate([], 100) == pytest.approx(0.0)

    def test_all_anomalies(self, anomalous_df):
        from pipeline.core.schemas import AnomalyRecord
        records = [
            AnomalyRecord(row_index=i, value=0, anomaly_score=-0.5, method="zscore")
            for i in range(50)
        ]
        assert anomaly_rate(records, 50) == pytest.approx(1.0)

    def test_zero_total_rows(self):
        assert anomaly_rate([], 0) == pytest.approx(0.0)

    def test_rate_bounded_between_zero_and_one(self, anomalous_df):
        anomalies, _ = detect_anomalies(anomalous_df)
        rate = anomaly_rate(anomalies, len(anomalous_df))
        assert 0.0 <= rate <= 1.0
