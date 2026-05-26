"""
scripts/run_evals.py
─────────────────────
Run the full eval harness against all 15 fixture tasks.

Usage:
    python scripts/run_evals.py
    python scripts/run_evals.py --parallel
    python scripts/run_evals.py --fixtures task_001 task_009
    python scripts/run_evals.py --category full_pipeline
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # root for evals/

from pipeline.middleware.logger import configure_logging
from pipeline.orchestrator.graph import build_graph
from evals.harness import load_fixtures, print_summary, run_eval_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pipeline eval harness")
    parser.add_argument(
        "--parallel", action="store_true",
        help="Run tasks concurrently (faster, more API calls)",
    )
    parser.add_argument(
        "--fixtures", nargs="+", metavar="TASK_ID",
        help="Run specific fixture IDs only (e.g. task_001 task_009)",
    )
    parser.add_argument(
        "--category", choices=["etl_only", "analysis_only", "full_pipeline", "edge_case"],
        help="Run only fixtures in this category",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    configure_logging(level=args.log_level, fmt="console")

    # Load and filter fixtures
    fixtures = load_fixtures()
    if not fixtures:
        print("❌ No fixtures found in evals/fixtures/. Run from project root.")
        sys.exit(1)

    if args.fixtures:
        fixtures = [f for f in fixtures if f.get("task_id") in args.fixtures]
        print(f"Filtered to {len(fixtures)} fixture(s): {args.fixtures}")

    if args.category:
        fixtures = [f for f in fixtures if f.get("category") == args.category]
        print(f"Filtered to {len(fixtures)} fixture(s) in category: {args.category}")

    if not fixtures:
        print("❌ No fixtures match the filter criteria.")
        sys.exit(1)

    # Build graph
    print(f"\nBuilding pipeline graph...")
    graph = build_graph()

    # Run harness
    print(f"Running {len(fixtures)} eval task(s) {'in parallel' if args.parallel else 'sequentially'}...\n")
    summary = await run_eval_harness(graph, fixtures, parallel=args.parallel)

    # Print results
    print_summary(summary)

    # Exit with non-zero code if pass rate is below 70%
    pass_rate = summary.passed_count / max(summary.total_tasks, 1)
    if pass_rate < 0.70:
        print(f"\n⚠️  Pass rate {pass_rate:.0%} below 70% threshold.")
        sys.exit(1)
    else:
        print(f"\n✅ Pass rate: {pass_rate:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
