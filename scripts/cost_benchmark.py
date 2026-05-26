"""
scripts/cost_benchmark.py
──────────────────────────
Benchmark cost across model tier scenarios.

Runs task_009 (full pipeline, clean data) 3 times under each scenario:
  - all_large:     all agents use claude-sonnet-4
  - mixed_routing: planner+orchestrator=sonnet, etl+analysis+report=haiku
  - all_small:     all agents use claude-haiku-4

Prints the cost ratio table and saves results to data/outputs/benchmark.json.

Usage:
    python scripts/cost_benchmark.py
    python scripts/cost_benchmark.py --runs 5
    python scripts/cost_benchmark.py --task task_001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # root for evals/

from pipeline.middleware.cost_model import BENCHMARK_SCENARIOS
from pipeline.middleware.logger import configure_logging
from pipeline.core.state import initial_state
from pipeline.core.schemas import CostSummary


BENCHMARK_TASK = (
    "Load data/samples/sales_monthly.csv, detect revenue anomalies, "
    "and generate a markdown report."
)
RESULTS_PATH = Path("data/outputs/benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model tier cost benchmark")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per scenario (default 3)")
    parser.add_argument("--task", default=None,
                        help="Override raw task string")
    parser.add_argument("--scenarios", nargs="+",
                        choices=list(BENCHMARK_SCENARIOS.keys()),
                        default=None,
                        help="Run specific scenarios only")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate cost without making API calls")
    return parser.parse_args()


async def run_scenario(
    scenario_name: str,
    model_map: dict[str, str],
    raw_task: str,
    n_runs: int,
) -> dict:
    """Run n_runs of the pipeline under the given model configuration."""
    from pipeline.orchestrator.graph import build_graph
    import os

    # Temporarily override model env vars
    original_env = {}
    env_map = {
        "planner":      "PLANNER_MODEL",
        "orchestrator": "ORCHESTRATOR_MODEL",
        "etl":          "ETL_AGENT_MODEL",
        "analysis":     "ANALYSIS_AGENT_MODEL",
        "report":       "REPORT_AGENT_MODEL",
    }
    for agent, model in model_map.items():
        env_key = env_map.get(agent)
        if env_key:
            original_env[env_key] = os.environ.get(env_key, "")
            os.environ[env_key] = model

    # Reload settings singleton with new env vars
    try:
        from pipeline.core import config as cfg_module
        cfg_module.get_settings.cache_clear()
    except Exception:
        pass

    graph = build_graph()
    run_results = []

    for run_i in range(n_runs):
        print(f"  [{scenario_name}] run {run_i+1}/{n_runs}...", end=" ", flush=True)
        start = time.monotonic()

        try:
            import uuid
            task_id = str(uuid.uuid4())
            start_state = initial_state(task_id=task_id, raw_task=raw_task)
            final_state = await graph.ainvoke(start_state)

            elapsed_ms = (time.monotonic() - start) * 1000
            cost_entries = final_state.get("cost_log", [])
            cost_sum = CostSummary.from_entries(task_id=task_id, entries=cost_entries)
            status = final_state.get("status", "unknown")

            run_results.append({
                "run": run_i + 1,
                "status": status.value if hasattr(status, "value") else str(status),
                "cost_usd": cost_sum.total_cost_usd,
                "input_tokens": cost_sum.total_input_tokens,
                "output_tokens": cost_sum.total_output_tokens,
                "latency_ms": round(elapsed_ms, 1),
                "per_agent": cost_sum.per_agent,
            })
            print(f"${cost_sum.total_cost_usd:.4f} ({elapsed_ms:.0f}ms)")

        except Exception as e:
            elapsed_ms = (time.monotonic() - start) * 1000
            run_results.append({
                "run": run_i + 1,
                "status": "failed",
                "cost_usd": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": round(elapsed_ms, 1),
                "error": str(e),
            })
            print(f"FAILED: {e}")

    # Restore env
    for env_key, val in original_env.items():
        if val:
            os.environ[env_key] = val
        else:
            os.environ.pop(env_key, None)

    # Aggregate
    successful = [r for r in run_results if r["status"] != "failed"]
    avg_cost = sum(r["cost_usd"] for r in successful) / max(len(successful), 1)
    avg_latency = sum(r["latency_ms"] for r in run_results) / max(len(run_results), 1)

    return {
        "scenario": scenario_name,
        "model_map": model_map,
        "n_runs": n_runs,
        "successful_runs": len(successful),
        "avg_cost_usd": round(avg_cost, 6),
        "avg_latency_ms": round(avg_latency, 1),
        "runs": run_results,
    }


def estimate_cost(scenario_name: str, model_map: dict[str, str]) -> dict:
    """
    Estimate cost without API calls using typical token counts.
    Used for --dry-run mode.
    """
    from pipeline.middleware.cost_model import compute_cost

    # Typical token usage per agent per run
    token_estimates = {
        "planner":      (600,  500),
        "orchestrator": (0,    0),    # no direct LLM call
        "etl":          (500,  300),
        "analysis":     (700,  500),
        "report":       (400,  250),
    }

    total_cost = 0.0
    per_agent = {}
    for agent, model in model_map.items():
        inp, out = token_estimates.get(agent, (0, 0))
        cost = compute_cost(model, inp, out)
        per_agent[agent] = round(cost, 6)
        total_cost += cost

    return {
        "scenario": scenario_name,
        "model_map": model_map,
        "estimated_cost_usd": round(total_cost, 6),
        "per_agent_estimate": per_agent,
        "note": "Dry run — estimated from typical token counts",
    }


def print_benchmark_table(results: list[dict]) -> None:
    """Print cost comparison table."""
    print("\n" + "─" * 65)
    print(f"{'Scenario':<20} {'Avg Cost USD':>14} {'Latency ms':>12} {'vs All-Large':>14}")
    print("─" * 65)

    baseline_cost = None
    for r in results:
        cost = r.get("avg_cost_usd") or r.get("estimated_cost_usd", 0)
        latency = r.get("avg_latency_ms", 0)

        if r["scenario"] == "all_large":
            baseline_cost = cost

        ratio = ""
        if baseline_cost and baseline_cost > 0 and cost > 0:
            multiplier = cost / baseline_cost
            ratio = f"{multiplier:.2f}×"

        print(f"  {r['scenario']:<18} ${cost:>12.4f}  {latency:>10.0f}  {ratio:>14}")

    if baseline_cost:
        mixed = next((r for r in results if r["scenario"] == "mixed_routing"), None)
        if mixed:
            mixed_cost = mixed.get("avg_cost_usd") or mixed.get("estimated_cost_usd", 0)
            if mixed_cost > 0:
                savings_pct = (1 - mixed_cost / baseline_cost) * 100
                print(f"\n  💡 Mixed routing saves ~{savings_pct:.0f}% vs all-large")
    print("─" * 65)


async def main() -> None:
    args = parse_args()
    configure_logging(level="INFO", fmt="console")

    raw_task = args.task or BENCHMARK_TASK
    scenarios_to_run = args.scenarios or list(BENCHMARK_SCENARIOS.keys())

    print(f"\n{'─'*55}")
    print(f"  Model Tier Cost Benchmark")
    print(f"  Task: {raw_task[:55]}...")
    print(f"  Runs per scenario: {args.runs}")
    print(f"  Scenarios: {', '.join(scenarios_to_run)}")
    print(f"  Mode: {'DRY RUN (estimate)' if args.dry_run else 'LIVE (real API calls)'}")
    print(f"{'─'*55}\n")

    results = []

    for scenario_name in scenarios_to_run:
        model_map = BENCHMARK_SCENARIOS[scenario_name]
        print(f"Scenario: {scenario_name}")
        for agent, model in model_map.items():
            print(f"  {agent}: {model}")

        if args.dry_run:
            result = estimate_cost(scenario_name, model_map)
            results.append(result)
            cost = result["estimated_cost_usd"]
            print(f"  → Estimated: ${cost:.4f} per run\n")
        else:
            result = await run_scenario(scenario_name, model_map, raw_task, args.runs)
            results.append(result)
            print()

    print_benchmark_table(results)

    # Save results
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
