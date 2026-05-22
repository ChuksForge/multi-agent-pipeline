"""
agents/report/tools/md_formatter.py
────────────────────────────────────
Markdown report formatter.

Assembles a structured Markdown report from typed result objects.
All section content is deterministic — derived from ETLResult,
AnalysisResult, and CostSummary. The executive summary is the only
LLM-generated section (written by the Report Agent before calling here).

Section order (fixed):
  1. Title + metadata
  2. Executive Summary      ← LLM-generated, passed in as string
  3. Data Overview          ← ETLResult row/col counts, source info
  4. Schema                 ← column names, dtypes, null rates
  5. Data Quality           ← ValidationIssue list
  6. Key Findings           ← AnalysisResult.key_findings bullets
  7. Statistical Summary    ← SummaryStats table per numeric column
  8. Anomaly Detection      ← AnomalyRecord table + rate
  9. Charts                 ← Vega-Lite specs as fenced JSON blocks
 10. Errors & Warnings      ← AgentErrorRecord list (only if present)
 11. Cost Summary           ← per-agent cost table
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    CostSummary,
    ETLResult,
    TaskPlan,
)
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)


def format_markdown(
    plan: TaskPlan,
    etl: ETLResult,
    analysis: AnalysisResult,
    executive_summary: str,
    cost_summary: CostSummary | None = None,
    errors: list[AgentErrorRecord] | None = None,
) -> str:
    """
    Assemble the full Markdown report from structured data.

    Args:
        plan:              TaskPlan (used for title and metadata).
        etl:               ETLResult from ETL Agent.
        analysis:          AnalysisResult from Analysis Agent.
        executive_summary: LLM-generated executive summary paragraph.
        cost_summary:      Optional cost breakdown.
        errors:            Optional list of AgentErrorRecords.

    Returns:
        Complete Markdown string.
    """
    sections: list[str] = []

    sections.append(_title_section(plan, etl, analysis))
    sections.append(_executive_summary_section(executive_summary))
    sections.append(_data_overview_section(etl))
    sections.append(_schema_section(etl))

    if etl.validation_issues:
        sections.append(_data_quality_section(etl))

    if analysis.key_findings:
        sections.append(_key_findings_section(analysis))

    if analysis.summary_stats:
        sections.append(_statistical_summary_section(analysis))

    sections.append(_anomaly_section(analysis, etl.row_count))

    if analysis.charts:
        sections.append(_charts_section(analysis))

    if errors:
        sections.append(_errors_section(errors))

    if cost_summary:
        sections.append(_cost_summary_section(cost_summary))

    report = "\n\n".join(s for s in sections if s.strip())

    logger.debug(
        "markdown_report_formatted",
        sections=len(sections),
        word_count=len(report.split()),
        charts=len(analysis.charts),
    )
    return report


# ── Section builders ──────────────────────────────────────────────────────────


def _title_section(plan: TaskPlan, etl: ETLResult, analysis: AnalysisResult) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status = "⚠️ Partial" if analysis.recovery_tier.value != "none" else "✅ Complete"
    return (
        f"# Data Pipeline Report\n\n"
        f"**Task:** {plan.raw_task}\n\n"
        f"**Generated:** {ts}  \n"
        f"**Task ID:** `{plan.task_id}`  \n"
        f"**Status:** {status}  \n"
        f"**Output format:** {plan.output_format.value.upper()}"
    )


def _executive_summary_section(summary: str) -> str:
    return f"## Executive Summary\n\n{summary.strip()}"


def _data_overview_section(etl: ETLResult) -> str:
    sources = ", ".join(f"`{sid}`" for sid in etl.source_ids) or "N/A"
    recovery = (
        "" if etl.recovery_tier.value == "none"
        else f"\n\n> ⚠️ **Recovery tier:** {etl.recovery_tier.value} — {'; '.join(etl.warnings)}"
    )
    return (
        f"## Data Overview\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Rows loaded | {etl.row_count:,} |\n"
        f"| Columns | {etl.column_count} |\n"
        f"| Sources | {sources} |\n"
        f"| ETL elapsed | {etl.elapsed_seconds:.2f}s |\n"
        f"| Validation errors | {sum(1 for i in etl.validation_issues if i.severity == 'error')} |\n"
        f"| Validation warnings | {sum(1 for i in etl.validation_issues if i.severity == 'warning')} |"
        f"{recovery}"
    )


def _schema_section(etl: ETLResult) -> str:
    if not etl.schema:
        return "## Schema\n\n_Schema information not available._"

    rows = ["## Schema\n", "| Column | Type | Nullable | Null Rate |", "|--------|------|----------|-----------|"]
    for col in etl.schema:
        null_rate_str = f"{col.null_rate:.1%}"
        nullable = "Yes" if col.nullable else "No"
        rows.append(f"| `{col.name}` | `{col.dtype}` | {nullable} | {null_rate_str} |")
    return "\n".join(rows)


def _data_quality_section(etl: ETLResult) -> str:
    errors = [i for i in etl.validation_issues if i.severity == "error"]
    warnings = [i for i in etl.validation_issues if i.severity == "warning"]

    lines = ["## Data Quality\n"]

    if errors:
        lines.append("### ❌ Errors\n")
        for issue in errors:
            col = f"`{issue.column}`" if issue.column else "Dataset"
            affected = f" ({issue.affected_rows:,} rows)" if issue.affected_rows else ""
            lines.append(f"- **{col}:** {issue.message}{affected}")
        lines.append("")

    if warnings:
        lines.append("### ⚠️ Warnings\n")
        for issue in warnings:
            col = f"`{issue.column}`" if issue.column else "Dataset"
            affected = f" ({issue.affected_rows:,} rows)" if issue.affected_rows else ""
            lines.append(f"- **{col}:** {issue.message}{affected}")

    return "\n".join(lines)


def _key_findings_section(analysis: AnalysisResult) -> str:
    lines = ["## Key Findings\n"]
    for finding in analysis.key_findings:
        lines.append(f"- {finding}")
    return "\n".join(lines)


def _statistical_summary_section(analysis: AnalysisResult) -> str:
    numeric = [s for s in analysis.summary_stats if s.mean is not None]
    categorical = [s for s in analysis.summary_stats if s.mean is None]

    lines = ["## Statistical Summary\n"]

    if numeric:
        lines.append("### Numeric Columns\n")
        lines.append("| Column | Count | Mean | Std | Min | Median | Max |")
        lines.append("|--------|-------|------|-----|-----|--------|-----|")
        for s in numeric:
            lines.append(
                f"| `{s.column}` "
                f"| {s.count:,} "
                f"| {s.mean:.2f} "
                f"| {s.std:.2f} "
                f"| {s.min:.2f} "
                f"| {s.p50:.2f} "
                f"| {s.max:.2f} |"
            )
        lines.append("")

    if categorical:
        lines.append("### Categorical Columns\n")
        lines.append("| Column | Count | Top Values |")
        lines.append("|--------|-------|------------|")
        for s in categorical:
            top = ", ".join(str(v) for v in s.top_values[:3]) or "N/A"
            lines.append(f"| `{s.column}` | {s.count:,} | {top} |")

    return "\n".join(lines)


def _anomaly_section(analysis: AnalysisResult, total_rows: int) -> str:
    rate_pct = f"{analysis.anomaly_rate:.2%}"
    recovery = (
        "" if analysis.recovery_tier.value == "none"
        else f"\n\n> ⚠️ **Detection note:** {analysis.recovery_tier.value} mode — "
             f"{'; '.join(analysis.warnings) if analysis.warnings else 'partial results'}"
    )

    if not analysis.anomalies:
        return (
            f"## Anomaly Detection\n\n"
            f"**Result:** No anomalies detected across {total_rows:,} rows.{recovery}"
        )

    lines = [
        f"## Anomaly Detection\n",
        f"**{analysis.anomaly_count} anomalies detected** "
        f"({rate_pct} of {total_rows:,} rows)\n",
        "| Row | Column | Value | Score | Method |",
        "|-----|--------|-------|-------|--------|",
    ]
    for a in analysis.anomalies[:20]:  # cap table at 20 rows
        val = str(a.value) if a.value is not None else "null"
        col = a.column or "N/A"
        lines.append(
            f"| {a.row_index} | `{col}` | {val} | {a.anomaly_score:.4f} | {a.method} |"
        )
    if len(analysis.anomalies) > 20:
        lines.append(f"\n_... and {len(analysis.anomalies) - 20} more anomalies not shown._")

    return "\n".join(lines) + recovery


def _charts_section(analysis: AnalysisResult) -> str:
    lines = ["## Charts\n"]
    lines.append(
        "_Charts are encoded as [Vega-Lite v5](https://vega.github.io/vega-lite/) "
        "specifications. Render with [vega-embed](https://github.com/vega/vega-embed) "
        "or paste into the [Vega Editor](https://vega.github.io/editor)._\n"
    )
    for chart in analysis.charts:
        lines.append(f"### {chart.title}\n")
        if chart.description:
            lines.append(f"_{chart.description}_\n")
        lines.append("```json")
        lines.append(json.dumps(chart.spec, indent=2))
        lines.append("```\n")
    return "\n".join(lines)


def _errors_section(errors: list[AgentErrorRecord]) -> str:
    lines = ["## Pipeline Errors\n"]
    lines.append(
        f"> ⚠️ The pipeline encountered {len(errors)} error(s) during execution. "
        "Results may be incomplete.\n"
    )
    lines.append("| Agent | Attempt | Error | Recovery |")
    lines.append("|-------|---------|-------|----------|")
    for e in errors:
        lines.append(
            f"| `{e.agent_id}` | {e.attempt} "
            f"| {e.error_type}: {e.message[:80]}{'...' if len(e.message) > 80 else ''} "
            f"| {e.recovery_tier.value} |"
        )
    return "\n".join(lines)


def _cost_summary_section(cost: CostSummary) -> str:
    lines = [
        "## Cost Summary\n",
        f"**Total cost:** ${cost.total_cost_usd:.6f} USD  \n"
        f"**Total tokens:** {cost.total_input_tokens + cost.total_output_tokens:,} "
        f"({cost.total_input_tokens:,} in / {cost.total_output_tokens:,} out)  \n"
        f"**Total latency:** {cost.total_latency_ms:.0f}ms\n",
        "### Cost by Agent\n",
        "| Agent | Cost (USD) |",
        "|-------|------------|",
    ]
    for agent, agent_cost in sorted(cost.per_agent.items(), key=lambda x: -x[1]):
        lines.append(f"| `{agent}` | ${agent_cost:.6f} |")

    lines.append("\n### Cost by Model\n")
    lines.append("| Model | Cost (USD) |")
    lines.append("|-------|------------|")
    for model, model_cost in sorted(cost.per_model.items(), key=lambda x: -x[1]):
        lines.append(f"| `{model}` | ${model_cost:.6f} |")

    return "\n".join(lines)


# ── Utility ───────────────────────────────────────────────────────────────────


def word_count(markdown: str) -> int:
    """Count words in rendered markdown (strips code blocks first)."""
    import re
    # Remove fenced code blocks
    clean = re.sub(r"```.*?```", "", markdown, flags=re.DOTALL)
    # Remove markdown syntax
    clean = re.sub(r"[#*`|_\-]", " ", clean)
    return len(clean.split())
