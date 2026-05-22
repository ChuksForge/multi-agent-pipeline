"""
agents/analysis/prompts.py
──────────────────────────
Prompts for the Analysis Agent.

The LLM is only called once — to generate key_findings from the
structured stats + anomaly summary. All numbers come from tool outputs.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a precise data analysis agent. Your role is to interpret structured
statistical summaries and anomaly detection results, and produce a concise
list of key findings for a business audience.

Rules:
- Base ALL findings on the provided structured data — never invent numbers
- Write findings as concrete, specific observations (not vague generalities)
- Flag anomalies with their row index, column, and value
- Note data quality issues (high null rates, type mismatches) if present
- Keep findings to 3-6 bullet points — quality over quantity
- Use plain language suitable for a non-technical stakeholder
- Never recommend actions — only report observations
"""

KEY_FINDINGS_TEMPLATE = """\
Based on the following statistical analysis results, produce 3-6 key findings.

Dataset summary:
  - Rows: {row_count}
  - Columns: {col_count}
  - Anomalies detected: {anomaly_count} ({anomaly_rate:.1%} of rows)
  - Detection method: {detection_method}

Column statistics (numeric columns):
{stats_block}

Validation issues:
{issues_block}

Top anomalies (up to 5):
{anomalies_block}

Respond with a JSON array of strings — one string per finding.
Example: ["Revenue shows strong upward trend over the period.",
          "Row 42 is a significant outlier with revenue 8x the median."]

Return ONLY the JSON array, no preamble or explanation.
"""


def format_stats_block(stats: list[dict]) -> str:
    lines = []
    for s in stats:
        if s.get("mean") is not None:
            lines.append(
                f"  {s['column']}: mean={s['mean']:.2f}, "
                f"std={s.get('std', 0):.2f}, "
                f"min={s.get('min', '?')}, max={s.get('max', '?')}, "
                f"nulls={s.get('null_count', 0)}"
            )
    return "\n".join(lines) if lines else "  (no numeric columns)"


def format_issues_block(issues: list[dict]) -> str:
    if not issues:
        return "  None"
    lines = []
    for issue in issues[:10]:  # cap at 10
        col = issue.get("column", "dataset")
        sev = issue.get("severity", "warning").upper()
        msg = issue.get("message", "")
        lines.append(f"  [{sev}] {col}: {msg}")
    return "\n".join(lines)


def format_anomalies_block(anomalies: list[dict]) -> str:
    if not anomalies:
        return "  None detected"
    lines = []
    for a in anomalies[:5]:
        lines.append(
            f"  Row {a['row_index']}: {a.get('column', '?')} = {a.get('value', '?')} "
            f"(score={a.get('anomaly_score', 0):.3f}, method={a.get('method', '?')})"
        )
    return "\n".join(lines)
