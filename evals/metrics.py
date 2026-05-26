"""
evals/metrics.py
─────────────────
Evaluation metrics for the multi-agent pipeline.

Three primary metrics:
  1. task_completion_rate  — fraction of runs reaching complete or partial
  2. output_correctness_score — weighted structural + content check
  3. cost_per_run          — delegates to CostSummary.from_entries()

All functions are pure — they take typed results and return numbers.
No side effects, no I/O. Called by harness.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from pipeline.core.schemas import (
    AnalysisResult,
    CostSummary,
    ETLResult,
    PipelineStatus,
    ReportResult,
)


# ─── Expected output schema ───────────────────────────────────────────────────


@dataclass
class ExpectedOutput:
    """
    Ground truth spec for a single eval fixture.
    All fields are optional — only specified fields are checked.
    """
    status: str = "complete"                    # complete | partial | failed
    min_rows: int = 0                           # ETL must load at least this many
    max_rows: int | None = None                 # ETL must not exceed this
    required_columns: list[str] = field(default_factory=list)
    max_anomaly_rate: float = 1.0               # analysis anomaly rate upper bound
    min_anomaly_count: int = 0                  # must detect at least N anomalies
    required_report_sections: list[str] = field(default_factory=list)
    min_word_count: int = 0
    allow_partial: bool = False                 # True = partial counts as pass


# ─── Metric 1 — Task Completion Rate ─────────────────────────────────────────


def task_completion_rate(statuses: list[PipelineStatus]) -> float:
    """
    Fraction of runs that completed (complete or partial).

    Args:
        statuses: List of final PipelineStatus from each run.

    Returns:
        Float in [0.0, 1.0].
    """
    if not statuses:
        return 0.0
    passing = sum(
        1 for s in statuses
        if s in (PipelineStatus.COMPLETE, PipelineStatus.PARTIAL)
    )
    return round(passing / len(statuses), 4)


# ─── Metric 2 — Output Correctness Score ─────────────────────────────────────


@dataclass
class CorrectnessResult:
    """Detailed correctness breakdown for one pipeline run."""
    task_id: str
    total_score: float          # 0.0 – 1.0
    schema_score: float         # 0.0 – 1.0  (weight 0.30)
    analysis_score: float       # 0.0 – 1.0  (weight 0.40)
    report_score: float         # 0.0 – 1.0  (weight 0.30)
    passed: bool
    failures: list[str] = field(default_factory=list)


def output_correctness_score(
    task_id: str,
    etl: ETLResult | None,
    analysis: AnalysisResult | None,
    report: ReportResult | None,
    expected: ExpectedOutput,
    pass_threshold: float = 0.60,
) -> CorrectnessResult:
    """
    Compute a weighted correctness score [0.0, 1.0] for one run.

    Weights:
      - Schema match (ETL):       0.30
      - Analysis correctness:     0.40
      - Report completeness:      0.30

    Args:
        task_id:        Pipeline task ID (for logging).
        etl:            ETLResult from the pipeline run.
        analysis:       AnalysisResult from the pipeline run.
        report:         ReportResult from the pipeline run.
        expected:       Ground truth ExpectedOutput spec.
        pass_threshold: Score at or above which the run is considered passing.

    Returns:
        CorrectnessResult with score breakdown and failure list.
    """
    failures: list[str] = []

    # ── Schema score (ETL) ────────────────────────────────────────────────────
    schema_score = _schema_score(etl, expected, failures)

    # ── Analysis score ────────────────────────────────────────────────────────
    analysis_score = _analysis_score(analysis, expected, failures)

    # ── Report score ──────────────────────────────────────────────────────────
    report_score = _report_score(report, expected, failures)

    # ── Weighted total ────────────────────────────────────────────────────────
    total = round(
        schema_score * 0.30
        + analysis_score * 0.40
        + report_score * 0.30,
        4,
    )

    return CorrectnessResult(
        task_id=task_id,
        total_score=total,
        schema_score=schema_score,
        analysis_score=analysis_score,
        report_score=report_score,
        passed=total >= pass_threshold,
        failures=failures,
    )


def _schema_score(
    etl: ETLResult | None,
    expected: ExpectedOutput,
    failures: list[str],
) -> float:
    if etl is None:
        failures.append("ETL result missing")
        return 0.0

    score = 1.0
    deductions = 0

    # Row count check
    if etl.row_count < expected.min_rows:
        failures.append(
            f"Row count {etl.row_count} < required {expected.min_rows}"
        )
        deductions += 1

    if expected.max_rows and etl.row_count > expected.max_rows:
        failures.append(
            f"Row count {etl.row_count} > max {expected.max_rows}"
        )
        deductions += 1

    # Required columns present
    actual_columns = {col.name for col in etl.schema}
    missing_cols = [c for c in expected.required_columns if c not in actual_columns]
    if missing_cols:
        failures.append(f"Missing required columns: {missing_cols}")
        deductions += len(missing_cols)

    # Score: deduct 0.25 per failure, floor at 0.0
    score = max(0.0, 1.0 - (deductions * 0.25))
    return round(score, 4)


def _analysis_score(
    analysis: AnalysisResult | None,
    expected: ExpectedOutput,
    failures: list[str],
) -> float:
    if analysis is None:
        failures.append("Analysis result missing")
        return 0.0

    score = 1.0
    deductions = 0

    # Anomaly rate within bounds
    if analysis.anomaly_rate > expected.max_anomaly_rate:
        failures.append(
            f"Anomaly rate {analysis.anomaly_rate:.2%} > max {expected.max_anomaly_rate:.2%}"
        )
        deductions += 1

    # Minimum anomaly count
    if analysis.anomaly_count < expected.min_anomaly_count:
        failures.append(
            f"Detected {analysis.anomaly_count} anomalies, expected >= {expected.min_anomaly_count}"
        )
        deductions += 1

    # Summary stats present
    if not analysis.summary_stats:
        failures.append("No summary statistics generated")
        deductions += 1

    score = max(0.0, 1.0 - (deductions * 0.33))
    return round(score, 4)


def _report_score(
    report: ReportResult | None,
    expected: ExpectedOutput,
    failures: list[str],
) -> float:
    if report is None:
        failures.append("Report result missing")
        return 0.0

    score = 1.0
    deductions = 0

    content = report.full_content or ""

    # Required sections present
    missing_sections = [
        s for s in expected.required_report_sections
        if s.lower() not in content.lower()
    ]
    if missing_sections:
        failures.append(f"Missing report sections: {missing_sections}")
        deductions += len(missing_sections)

    # Minimum word count
    if report.word_count < expected.min_word_count:
        failures.append(
            f"Report word count {report.word_count} < required {expected.min_word_count}"
        )
        deductions += 1

    # Report is not empty
    if len(content.strip()) < 50:
        failures.append("Report content appears empty or too short")
        deductions += 2

    score = max(0.0, 1.0 - (deductions * 0.25))
    return round(score, 4)


# ─── Metric 3 — Cost Per Run ──────────────────────────────────────────────────


def cost_per_run(
    task_id: str,
    cost_entries: list,
) -> CostSummary:
    """
    Build a CostSummary from raw CostEntry list.
    Delegates to CostSummary.from_entries().
    """
    return CostSummary.from_entries(task_id=task_id, entries=cost_entries)


# ─── Aggregate reporting ──────────────────────────────────────────────────────


@dataclass
class EvalSummary:
    """Aggregated results across all eval fixtures."""
    total_tasks: int
    completion_rate: float
    avg_correctness: float
    avg_cost_usd: float
    avg_latency_ms: float
    passed_count: int
    failed_count: int
    partial_count: int
    per_task: list[dict[str, Any]] = field(default_factory=list)


def aggregate_results(results: list[dict[str, Any]]) -> EvalSummary:
    """
    Aggregate a list of per-task result dicts into an EvalSummary.

    Each result dict must have keys:
      status, correctness_score, cost_usd, latency_ms, passed
    """
    if not results:
        return EvalSummary(
            total_tasks=0, completion_rate=0.0, avg_correctness=0.0,
            avg_cost_usd=0.0, avg_latency_ms=0.0,
            passed_count=0, failed_count=0, partial_count=0,
        )

    statuses = [PipelineStatus(r.get("status", "failed")) for r in results]
    completion = task_completion_rate(statuses)

    correctness_scores = [r.get("correctness_score", 0.0) for r in results]
    avg_correctness = round(sum(correctness_scores) / len(correctness_scores), 4)

    costs = [r.get("cost_usd", 0.0) for r in results]
    avg_cost = round(sum(costs) / len(costs), 6)

    latencies = [r.get("latency_ms", 0.0) for r in results]
    avg_latency = round(sum(latencies) / len(latencies), 1)

    passed = sum(1 for r in results if r.get("passed", False))
    failed = sum(1 for r in results if r.get("status") == "failed")
    partial = sum(1 for r in results if r.get("status") == "partial")

    return EvalSummary(
        total_tasks=len(results),
        completion_rate=completion,
        avg_correctness=avg_correctness,
        avg_cost_usd=avg_cost,
        avg_latency_ms=avg_latency,
        passed_count=passed,
        failed_count=failed,
        partial_count=partial,
        per_task=results,
    )
