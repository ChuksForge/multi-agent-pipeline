"""
agents/report/prompts.py
────────────────────────
Prompts for the Report Agent.

One LLM call only — to generate the executive summary paragraph.
Everything else in the report is deterministic from structured data.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a professional data analyst writing an executive summary for a
technical data pipeline report. Your audience is a mixed group of
engineers and business stakeholders.

Rules:
- Write exactly ONE paragraph (4-7 sentences)
- Lead with the most important finding
- Include the data size (rows and columns)
- Mention anomaly count and rate if any were detected
- Note any data quality issues briefly
- Close with a statement on report completeness
- Use plain, direct language — no jargon, no filler phrases
- Do not use bullet points or headers — prose only
- Never invent numbers — use only those provided
"""

EXECUTIVE_SUMMARY_TEMPLATE = """\
Write an executive summary paragraph for a data pipeline report with these results:

Dataset: {row_count:,} rows × {col_count} columns
Sources: {sources}
Task: {raw_task}

Analysis results:
- Anomalies detected: {anomaly_count} ({anomaly_rate:.1%} of rows)
- Detection method: IsolationForest + Z-score ensemble
- Key finding: {top_finding}

Data quality:
- Validation errors: {error_count}
- Validation warnings: {warning_count}

Pipeline status: {status}
{recovery_note}

Write a single paragraph executive summary.
"""

FALLBACK_SUMMARY_TEMPLATE = (
    "This report covers a dataset of {row_count:,} rows and {col_count} columns "
    "loaded from {source_count} source(s). "
    "{anomaly_sentence}"
    "{quality_sentence}"
    "The pipeline completed with status: {status}."
)


def format_executive_summary_prompt(
    row_count: int,
    col_count: int,
    sources: list[str],
    raw_task: str,
    anomaly_count: int,
    anomaly_rate: float,
    error_count: int,
    warning_count: int,
    status: str,
    key_findings: list[str],
    recovery_tier: str,
) -> str:
    top_finding = key_findings[0] if key_findings else "No specific findings available."
    recovery_note = (
        f"Note: Pipeline ran in {recovery_tier} mode — results may be partial."
        if recovery_tier != "none" else ""
    )
    source_list = ", ".join(sources) if sources else "unknown"
    return EXECUTIVE_SUMMARY_TEMPLATE.format(
        row_count=row_count,
        col_count=col_count,
        sources=source_list,
        raw_task=raw_task,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_rate,
        top_finding=top_finding,
        error_count=error_count,
        warning_count=warning_count,
        status=status,
        recovery_note=recovery_note,
    )


def build_fallback_summary(
    row_count: int,
    col_count: int,
    source_count: int,
    anomaly_count: int,
    anomaly_rate: float,
    error_count: int,
    status: str,
) -> str:
    """Deterministic summary used when the LLM call fails."""
    if anomaly_count > 0:
        anomaly_sentence = (
            f"Anomaly detection identified {anomaly_count} anomalies "
            f"({anomaly_rate:.1%} of rows) using ensemble methods. "
        )
    else:
        anomaly_sentence = "No anomalies were detected in the dataset. "

    if error_count > 0:
        quality_sentence = (
            f"Data quality checks flagged {error_count} error(s) that may affect results. "
        )
    else:
        quality_sentence = "All data quality checks passed without errors. "

    return FALLBACK_SUMMARY_TEMPLATE.format(
        row_count=row_count,
        col_count=col_count,
        source_count=source_count,
        anomaly_sentence=anomaly_sentence,
        quality_sentence=quality_sentence,
        status=status,
    )
