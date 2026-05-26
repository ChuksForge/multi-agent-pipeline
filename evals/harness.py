"""
evals/harness.py
─────────────────
Eval harness — runs all fixture tasks and measures pipeline quality.

Usage:
    python scripts/run_evals.py

Design:
  - Loads all fixtures from evals/fixtures/*.json
  - Runs each through the compiled LangGraph pipeline
  - Records status, correctness score, cost, and latency per task
  - Writes results to data/outputs/eval_results.jsonl (one JSON per line)
  - Prints a rich summary table at the end

Concurrency:
  - Runs tasks sequentially by default (safe for rate-limited APIs)
  - Set EVAL_PARALLEL=1 in env to run with asyncio.gather (faster, more API calls)

Fixtures format (evals/fixtures/task_NNN_*.json):
{
  "task_id": "task_001",
  "raw_task": "Load and validate the sales CSV",
  "data_source": "data/samples/sales_monthly.csv",
  "expected": {
    "status": "complete",
    "min_rows": 100,
    "required_columns": ["date", "revenue"],
    "max_anomaly_rate": 0.10
  }
}
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from pipeline.core.schemas import (
    CostSummary,
    PipelineStatus,
)
from pipeline.core.state import PipelineState
from evals.metrics import (
    ExpectedOutput,
    EvalSummary,
    aggregate_results,
    cost_per_run,
    output_correctness_score,
)
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

FIXTURES_DIR = Path("evals/fixtures")
RESULTS_PATH = Path("data/outputs/eval_results.jsonl")


# ─── Fixture loading ──────────────────────────────────────────────────────────


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> list[dict[str, Any]]:
    """
    Load all JSON fixture files from the fixtures directory.
    Returns sorted list of fixture dicts.
    """
    if not fixtures_dir.exists():
        logger.warning("fixtures_dir_missing", path=str(fixtures_dir))
        return []

    fixtures = []
    for path in sorted(fixtures_dir.glob("task_*.json")):
        try:
            fixture = json.loads(path.read_text())
            fixture["_fixture_path"] = str(path)
            fixtures.append(fixture)
        except Exception as e:
            logger.warning("fixture_load_failed", path=str(path), error=str(e))

    logger.info("fixtures_loaded", count=len(fixtures))
    return fixtures


def parse_expected(fixture: dict[str, Any]) -> ExpectedOutput:
    """Build an ExpectedOutput from a fixture's 'expected' block."""
    exp = fixture.get("expected", {})
    return ExpectedOutput(
        status=exp.get("status", "complete"),
        min_rows=exp.get("min_rows", 0),
        max_rows=exp.get("max_rows"),
        required_columns=exp.get("required_columns", []),
        max_anomaly_rate=exp.get("max_anomaly_rate", 1.0),
        min_anomaly_count=exp.get("min_anomaly_count", 0),
        required_report_sections=exp.get("required_report_sections", []),
        min_word_count=exp.get("min_word_count", 0),
        allow_partial=exp.get("allow_partial", False),
    )


# ─── Single task runner ───────────────────────────────────────────────────────


async def run_single_task(
    graph: Any,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    """
    Run one fixture task through the pipeline and return a result dict.

    Args:
        graph:   Compiled LangGraph pipeline.
        fixture: Fixture definition dict.

    Returns:
        Dict with keys: task_id, status, correctness_score, cost_usd,
        latency_ms, passed, failures, error (if crashed).
    """
    task_id = fixture.get("task_id", "unknown")
    raw_task = fixture.get("raw_task", "")
    expected = parse_expected(fixture)

    # Inject data source path into task string if provided
    data_source = fixture.get("data_source", "")
    if data_source and data_source not in raw_task:
        raw_task = f"{raw_task} Data file: {data_source}"

    logger.info("eval_task_start", task_id=task_id)
    start = time.monotonic()

    try:
        from pipeline.core.state import initial_state
        start_state = initial_state(task_id=task_id, raw_task=raw_task)
        final_state: PipelineState = await graph.ainvoke(start_state)

        elapsed_ms = (time.monotonic() - start) * 1000
        status = final_state.get("status", PipelineStatus.FAILED)

        # Score correctness
        correctness = output_correctness_score(
            task_id=task_id,
            etl=final_state.get("etl_result"),
            analysis=final_state.get("analysis_result"),
            report=final_state.get("report_result"),
            expected=expected,
        )

        # Cost summary
        cost_entries = final_state.get("cost_log", [])
        cost_sum = cost_per_run(task_id, cost_entries)

        result = {
            "task_id": task_id,
            "raw_task": raw_task[:80],
            "status": status.value,
            "correctness_score": correctness.total_score,
            "schema_score": correctness.schema_score,
            "analysis_score": correctness.analysis_score,
            "report_score": correctness.report_score,
            "passed": correctness.passed,
            "failures": correctness.failures,
            "cost_usd": cost_sum.total_cost_usd,
            "total_tokens": cost_sum.total_input_tokens + cost_sum.total_output_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "anomaly_count": final_state.get("analysis_result", None) and
                             final_state["analysis_result"].anomaly_count or 0,
            "row_count": final_state.get("etl_result", None) and
                         final_state["etl_result"].row_count or 0,
        }

        logger.info(
            "eval_task_complete",
            task_id=task_id,
            status=status.value,
            score=correctness.total_score,
            passed=correctness.passed,
            cost_usd=cost_sum.total_cost_usd,
            latency_ms=round(elapsed_ms, 1),
        )
        return result

    except Exception as e:
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.error("eval_task_crashed", task_id=task_id, error=str(e))
        return {
            "task_id": task_id,
            "raw_task": raw_task[:80],
            "status": "failed",
            "correctness_score": 0.0,
            "schema_score": 0.0,
            "analysis_score": 0.0,
            "report_score": 0.0,
            "passed": False,
            "failures": [f"Pipeline crashed: {e}"],
            "cost_usd": 0.0,
            "total_tokens": 0,
            "latency_ms": round(elapsed_ms, 1),
            "anomaly_count": 0,
            "row_count": 0,
            "error": str(e),
        }


# ─── Main harness ─────────────────────────────────────────────────────────────


async def run_eval_harness(
    graph: Any,
    fixtures: list[dict[str, Any]],
    parallel: bool = False,
) -> EvalSummary:
    """
    Run all fixtures and return aggregated EvalSummary.

    Args:
        graph:    Compiled LangGraph pipeline.
        fixtures: List of fixture dicts from load_fixtures().
        parallel: If True, run all tasks concurrently via asyncio.gather.
                  Warning: may hit API rate limits.

    Returns:
        EvalSummary with per-task and aggregate metrics.
    """
    if not fixtures:
        logger.warning("no_fixtures_found")
        return aggregate_results([])

    logger.info("eval_harness_start", tasks=len(fixtures), parallel=parallel)

    if parallel:
        results = await asyncio.gather(
            *[run_single_task(graph, f) for f in fixtures],
            return_exceptions=False,
        )
        results = list(results)
    else:
        results = []
        for fixture in fixtures:
            result = await run_single_task(graph, fixture)
            results.append(result)

    # Write JSONL results file
    _write_results(results)

    summary = aggregate_results(results)
    logger.info(
        "eval_harness_complete",
        total=summary.total_tasks,
        completion_rate=summary.completion_rate,
        avg_correctness=summary.avg_correctness,
        passed=summary.passed_count,
        avg_cost_usd=summary.avg_cost_usd,
    )
    return summary


def _write_results(results: list[dict[str, Any]]) -> None:
    """Write results to JSONL file, one JSON object per line."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    logger.info("results_written", path=str(RESULTS_PATH), count=len(results))


# ─── Console reporter ─────────────────────────────────────────────────────────


def print_summary(summary: EvalSummary) -> None:
    """Print a formatted summary table to stdout using rich."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print("\n[bold cyan]── Eval Results ──────────────────────────────[/bold cyan]")
        console.print(f"  Tasks run:        [bold]{summary.total_tasks}[/bold]")
        console.print(f"  Completion rate:  [bold green]{summary.completion_rate:.1%}[/bold green]")
        console.print(f"  Avg correctness:  [bold]{summary.avg_correctness:.3f}[/bold]")
        console.print(f"  Passed:           [bold green]{summary.passed_count}[/bold green] / {summary.total_tasks}")
        console.print(f"  Failed:           [bold red]{summary.failed_count}[/bold red]")
        console.print(f"  Partial:          [yellow]{summary.partial_count}[/yellow]")
        console.print(f"  Avg cost/run:     ${summary.avg_cost_usd:.4f} USD")
        console.print(f"  Avg latency:      {summary.avg_latency_ms:.0f}ms\n")

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("Task ID", style="cyan")
        table.add_column("Status")
        table.add_column("Score", justify="right")
        table.add_column("Pass")
        table.add_column("Cost USD", justify="right")
        table.add_column("ms", justify="right")

        for r in summary.per_task:
            status_color = {"complete": "green", "partial": "yellow", "failed": "red"}.get(
                r.get("status", "failed"), "white"
            )
            passed_icon = "✅" if r.get("passed") else "❌"
            table.add_row(
                r.get("task_id", "?"),
                f"[{status_color}]{r.get('status', '?')}[/{status_color}]",
                f"{r.get('correctness_score', 0):.3f}",
                passed_icon,
                f"${r.get('cost_usd', 0):.4f}",
                f"{r.get('latency_ms', 0):.0f}",
            )

        console.print(table)

    except ImportError:
        # Fallback: plain print
        print(f"\n── Eval Results ──────────────────────────")
        print(f"  Tasks:       {summary.total_tasks}")
        print(f"  Completion:  {summary.completion_rate:.1%}")
        print(f"  Avg score:   {summary.avg_correctness:.3f}")
        print(f"  Passed:      {summary.passed_count}/{summary.total_tasks}")
        print(f"  Avg cost:    ${summary.avg_cost_usd:.4f}")
        print(f"  Avg latency: {summary.avg_latency_ms:.0f}ms")
        for r in summary.per_task:
            icon = "✓" if r.get("passed") else "✗"
            print(f"  [{icon}] {r['task_id']}: {r.get('status')} "
                  f"score={r.get('correctness_score', 0):.3f} "
                  f"cost=${r.get('cost_usd', 0):.4f}")
