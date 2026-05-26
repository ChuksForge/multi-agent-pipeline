"""
scripts/run_demo.py
────────────────────
One-command demo: runs the full pipeline on sales_monthly.csv
and prints the Markdown report + cost summary to the terminal.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --task "Analyse server_metrics.parquet for CPU anomalies"
    python scripts/run_demo.py --format json
    python scripts/run_demo.py --quiet   (suppress progress output)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.middleware.logger import configure_logging
from pipeline.core.schemas import CostSummary, PipelineStatus


DEFAULT_TASK = (
    "Load data/samples/sales_monthly.csv, detect revenue anomalies "
    "using ensemble methods, and generate a markdown report."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Data Pipeline Demo")
    parser.add_argument("--task", default=DEFAULT_TASK,
                        help="Natural language task description")
    parser.add_argument("--format", choices=["markdown", "json", "pdf"],
                        default="markdown", dest="output_format")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output, show only the report")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, fmt="console")

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.table import Table
        from rich import box
        rich_available = True
        console = Console()
    except ImportError:
        rich_available = False
        console = None

    def print_step(msg: str) -> None:
        if not args.quiet:
            if rich_available:
                console.print(f"  [dim]→[/dim] {msg}")
            else:
                print(f"  → {msg}")

    if not args.quiet:
        if rich_available:
            console.print(Panel.fit(
                "[bold cyan]Multi-Agent Data Pipeline[/bold cyan]\n"
                "[dim]ETL → Analysis → Report · LangGraph · Claude[/dim]",
                border_style="cyan",
            ))
            console.print(f"\n[bold]Task:[/bold] {args.task}\n")
        else:
            print("\n── Multi-Agent Data Pipeline ──────────────────")
            print(f"Task: {args.task}\n")

    # Build and run pipeline
    print_step("Building pipeline graph...")
    from pipeline.orchestrator.graph import build_graph, run_pipeline_sync

    graph = build_graph()

    print_step("Running pipeline (ETL → Analysis → Report)...")
    start = time.monotonic()

    raw_task = args.task
    if args.output_format != "markdown":
        raw_task += f" Output format: {args.output_format}."

    final_state = run_pipeline_sync(graph, raw_task=raw_task)
    elapsed = time.monotonic() - start

    # Extract results
    status = final_state.get("status", PipelineStatus.FAILED)
    etl = final_state.get("etl_result")
    analysis = final_state.get("analysis_result")
    report = final_state.get("report_result")
    cost_entries = final_state.get("cost_log", [])

    # Status line
    status_label = {
        PipelineStatus.COMPLETE: "✅ Complete",
        PipelineStatus.PARTIAL:  "⚠️  Partial",
        PipelineStatus.FAILED:   "❌ Failed",
    }.get(status, str(status))

    if not args.quiet:
        print()
        if rich_available:
            console.print(f"[bold]Status:[/bold]  {status_label}  "
                          f"[dim]({elapsed:.1f}s wall clock)[/dim]")
            if etl:
                console.print(f"[bold]ETL:[/bold]     {etl.row_count:,} rows · "
                               f"{etl.column_count} columns · "
                               f"{len(etl.validation_issues)} issues")
            if analysis:
                console.print(f"[bold]Analysis:[/bold] {analysis.anomaly_count} anomalies "
                               f"({analysis.anomaly_rate:.1%}) · "
                               f"{len(analysis.charts)} charts")
        else:
            print(f"Status:  {status_label} ({elapsed:.1f}s)")
            if etl:
                print(f"ETL:     {etl.row_count:,} rows · {etl.column_count} columns")
            if analysis:
                print(f"Analysis: {analysis.anomaly_count} anomalies ({analysis.anomaly_rate:.1%})")
        print()

    # Print the report
    if report and report.full_content:
        if not args.quiet:
            separator = "─" * 60
            print(separator)
            print("  REPORT OUTPUT")
            print(separator)

        if rich_available and args.output_format == "markdown":
            console.print(Markdown(report.full_content))
        else:
            print(report.full_content)
    else:
        print("⚠️  No report generated.")

    # Cost summary table
    if cost_entries and not args.quiet:
        cost_sum = CostSummary.from_entries(
            task_id=final_state.get("task_id", "demo"),
            entries=cost_entries,
        )
        if rich_available:
            table = Table(title="Cost Summary", box=box.SIMPLE, show_header=True,
                          header_style="bold")
            table.add_column("Agent", style="cyan")
            table.add_column("Cost USD", justify="right")
            for agent, cost in sorted(cost_sum.per_agent.items(), key=lambda x: -x[1]):
                table.add_row(agent, f"${cost:.6f}")
            table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]${cost_sum.total_cost_usd:.6f}[/bold]",
            )
            console.print()
            console.print(table)
            console.print(
                f"  Tokens: {cost_sum.total_input_tokens:,} in / "
                f"{cost_sum.total_output_tokens:,} out\n"
            )
        else:
            print(f"\n── Cost Summary ──────────────────")
            for agent, cost in sorted(cost_sum.per_agent.items(), key=lambda x: -x[1]):
                print(f"  {agent}: ${cost:.6f}")
            print(f"  TOTAL:  ${cost_sum.total_cost_usd:.6f}")

    # Exit code
    sys.exit(0 if status in (PipelineStatus.COMPLETE, PipelineStatus.PARTIAL) else 1)


if __name__ == "__main__":
    asyncio.run(main())
