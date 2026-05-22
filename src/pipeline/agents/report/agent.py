"""
agents/report/agent.py
───────────────────────
Report Agent — LangGraph node.

Responsibilities:
  1. Generate executive summary via one LLM call (Haiku)
  2. Assemble full report via md_formatter or json_emitter
  3. Optionally render PDF via pdf_renderer
  4. Write ReportResult into PipelineState

Recovery tiers:
  NONE       — clean run, full Markdown/JSON/PDF produced
  SIMPLIFIED — Markdown failed, fell back to JSON
  DEGRADED   — both failed, returned raw structured data dump

One LLM call only — for the executive summary paragraph.
All other content is assembled deterministically from typed results.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_anthropic import ChatAnthropic

from pipeline.agents.report.prompts import (
    SYSTEM_PROMPT,
    build_fallback_summary,
    format_executive_summary_prompt,
)
from pipeline.agents.report.tools.json_emitter import emit_json
from pipeline.agents.report.tools.md_formatter import format_markdown, word_count
from pipeline.agents.report.tools.pdf_renderer import is_available as pdf_available
from pipeline.agents.report.tools.pdf_renderer import render_pdf
from pipeline.core.config import settings
from pipeline.core.schemas import (
    AgentErrorRecord,
    AnalysisResult,
    CostSummary,
    ETLResult,
    OutputFormat,
    PipelineStatus,
    RecoveryTier,
    ReportResult,
    ReportSection,
    TaskPlan,
)
from pipeline.core.state import PipelineState
from pipeline.middleware.logger import bind_pipeline_context, get_logger
from pipeline.middleware.token_tracker import make_tracker

logger = get_logger(__name__)

AGENT_ID = "report"


# ─── LangGraph Node ───────────────────────────────────────────────────────────


def report_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node function for the Report Agent.
    Receives full PipelineState, returns partial update dict.
    Never raises — all errors captured in return dict.
    """
    task_id = state["task_id"]
    bind_pipeline_context(task_id=task_id, agent_id=AGENT_ID)
    tracker = make_tracker(task_id=task_id, agent_id=AGENT_ID)
    start = time.monotonic()

    logger.info("report_agent_start", task_id=task_id)

    # ── Collect inputs from state ─────────────────────────────────────────────
    task_plan: TaskPlan | None = state.get("task_plan")
    etl_result: ETLResult | None = state.get("etl_result")
    analysis_result: AnalysisResult | None = state.get("analysis_result")
    prior_errors: list[AgentErrorRecord] = list(state.get("errors", []))
    cost_entries = list(state.get("cost_log", []))

    if task_plan is None:
        return _degraded_update(
            task_id=task_id,
            message="No TaskPlan available — cannot determine output format",
            elapsed=time.monotonic() - start,
            tracker_entries=tracker.entries,
        )

    # Use empty results if agents degraded — report errors instead
    etl = etl_result or _empty_etl(task_id)
    analysis = analysis_result or _empty_analysis(task_id)

    output_format = task_plan.output_format
    cost_summary = (
        CostSummary.from_entries(task_id=task_id, entries=cost_entries)
        if cost_entries else None
    )

    # ── Generate executive summary (one LLM call) ─────────────────────────────
    executive_summary = _generate_executive_summary(
        plan=task_plan,
        etl=etl,
        analysis=analysis,
        tracker=tracker,
    )

    # ── Assemble report ───────────────────────────────────────────────────────
    try:
        result = _assemble_report(
            task_id=task_id,
            plan=task_plan,
            etl=etl,
            analysis=analysis,
            executive_summary=executive_summary,
            cost_summary=cost_summary,
            errors=prior_errors if prior_errors else None,
            output_format=output_format,
            tracker=tracker,
        )

        elapsed = time.monotonic() - start
        result.elapsed_seconds = round(elapsed, 3)

        logger.info(
            "report_agent_complete",
            output_format=output_format.value,
            word_count=result.word_count,
            recovery_tier=result.recovery_tier.value,
            elapsed_s=round(elapsed, 2),
        )

        return {
            "report_result": result,
            "cost_log": tracker.entries,
            "current_agent": None,
            "status": PipelineStatus.COMPLETE
            if result.recovery_tier == RecoveryTier.NONE
            else PipelineStatus.PARTIAL,
        }

    # ── Tier 2: JSON fallback ─────────────────────────────────────────────────
    except Exception as e:
        logger.warning("report_markdown_failed", error=str(e), fallback="json")
        try:
            json_content = emit_json(
                plan=task_plan,
                etl=etl,
                analysis=analysis,
                executive_summary=executive_summary,
                cost_summary=cost_summary,
                errors=prior_errors if prior_errors else None,
            )
            elapsed = time.monotonic() - start
            result = ReportResult(
                task_id=task_id,
                output_format=OutputFormat.JSON,
                title="Data Pipeline Report (JSON fallback)",
                full_content=json_content,
                word_count=len(json_content.split()),
                elapsed_seconds=round(elapsed, 3),
                recovery_tier=RecoveryTier.SIMPLIFIED,
            )
            return {
                "report_result": result,
                "cost_log": tracker.entries,
                "current_agent": None,
                "status": PipelineStatus.PARTIAL,
            }

        # ── Tier 3: raw dump ──────────────────────────────────────────────────
        except Exception as fallback_err:
            return _degraded_update(
                task_id=task_id,
                message=f"Report generation fully failed: {fallback_err}",
                elapsed=time.monotonic() - start,
                tracker_entries=tracker.entries,
            )


# ─── Core Report Logic ────────────────────────────────────────────────────────


def _assemble_report(
    task_id: str,
    plan: TaskPlan,
    etl: ETLResult,
    analysis: AnalysisResult,
    executive_summary: str,
    cost_summary: CostSummary | None,
    errors: list[AgentErrorRecord] | None,
    output_format: OutputFormat,
    tracker: Any,
) -> ReportResult:
    """Assemble the full report in the requested output format."""

    title = f"Data Pipeline Report — {plan.raw_task[:60]}"

    if output_format == OutputFormat.JSON:
        content = emit_json(
            plan=plan, etl=etl, analysis=analysis,
            executive_summary=executive_summary,
            cost_summary=cost_summary, errors=errors,
        )
        return ReportResult(
            task_id=task_id,
            output_format=OutputFormat.JSON,
            title=title,
            full_content=content,
            word_count=len(content.split()),
            recovery_tier=RecoveryTier.NONE,
        )

    # Markdown (default) — also used as intermediate for PDF
    md_content = format_markdown(
        plan=plan, etl=etl, analysis=analysis,
        executive_summary=executive_summary,
        cost_summary=cost_summary, errors=errors,
    )

    # Build section list for ReportResult
    sections = _extract_sections(md_content)

    if output_format == OutputFormat.PDF:
        output_path = _pdf_output_path(task_id)
        if pdf_available():
            try:
                render_pdf(md_content, output_path)
                logger.info("pdf_written", path=output_path)
            except Exception as e:
                logger.warning("pdf_render_failed", error=str(e), fallback="markdown")
        else:
            logger.warning(
                "weasyprint_not_installed",
                note="Returning Markdown content; install [pdf] extras for PDF output",
            )
        return ReportResult(
            task_id=task_id,
            output_format=OutputFormat.PDF,
            title=title,
            sections=sections,
            full_content=md_content,  # Markdown as fallback content
            output_path=output_path if pdf_available() else None,
            word_count=word_count(md_content),
            recovery_tier=RecoveryTier.NONE if pdf_available() else RecoveryTier.SIMPLIFIED,
        )

    # Default: Markdown
    return ReportResult(
        task_id=task_id,
        output_format=OutputFormat.MARKDOWN,
        title=title,
        sections=sections,
        full_content=md_content,
        word_count=word_count(md_content),
        recovery_tier=RecoveryTier.NONE,
    )


def _generate_executive_summary(
    plan: TaskPlan,
    etl: ETLResult,
    analysis: AnalysisResult,
    tracker: Any,
) -> str:
    """One Haiku call — returns the executive summary paragraph."""
    try:
        prompt = format_executive_summary_prompt(
            row_count=etl.row_count,
            col_count=etl.column_count,
            sources=etl.source_ids,
            raw_task=plan.raw_task,
            anomaly_count=analysis.anomaly_count,
            anomaly_rate=analysis.anomaly_rate,
            error_count=sum(1 for i in etl.validation_issues if i.severity == "error"),
            warning_count=sum(1 for i in etl.validation_issues if i.severity == "warning"),
            status="complete" if analysis.recovery_tier.value == "none" else "partial",
            key_findings=analysis.key_findings,
            recovery_tier=analysis.recovery_tier.value,
        )

        llm = ChatAnthropic(
            model=settings.report_agent_model,
            max_tokens=256,
            callbacks=[tracker],
        )
        response = llm.invoke([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in content
            )
        return content.strip()

    except Exception as e:
        logger.warning("executive_summary_llm_failed", error=str(e))
        return build_fallback_summary(
            row_count=etl.row_count,
            col_count=etl.column_count,
            source_count=len(etl.source_ids),
            anomaly_count=analysis.anomaly_count,
            anomaly_rate=analysis.anomaly_rate,
            error_count=sum(1 for i in etl.validation_issues if i.severity == "error"),
            status="partial" if analysis.recovery_tier.value != "none" else "complete",
        )


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _extract_sections(markdown: str) -> list[ReportSection]:
    """Split Markdown by H2 headings into ReportSection list."""
    import re
    parts = re.split(r"\n## ", markdown)
    sections = []
    for i, part in enumerate(parts):
        if i == 0:
            title = "Header"
            content = part
        else:
            lines = part.split("\n", 1)
            title = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""
        sections.append(ReportSection(title=title, content=content.strip(), order=i))
    return sections


def _pdf_output_path(task_id: str) -> str:
    output_dir = Path(settings.outputs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir / f"report_{task_id}.pdf")


def _empty_etl(task_id: str) -> ETLResult:
    """Minimal ETLResult for when the ETL agent degraded."""
    from pipeline.core.schemas import ETLResult
    return ETLResult(
        task_id=task_id,
        source_ids=[],
        row_count=0,
        column_count=0,
        schema=[],
        recovery_tier=RecoveryTier.DEGRADED,
        warnings=["ETL result not available"],
    )


def _empty_analysis(task_id: str) -> AnalysisResult:
    """Minimal AnalysisResult for when the Analysis agent degraded."""
    from pipeline.core.schemas import AnalysisResult
    return AnalysisResult(
        task_id=task_id,
        recovery_tier=RecoveryTier.DEGRADED,
        warnings=["Analysis result not available"],
    )


def _degraded_update(
    task_id: str,
    message: str,
    elapsed: float,
    tracker_entries: list,
) -> dict[str, Any]:
    error_record = AgentErrorRecord(
        agent_id=AGENT_ID,
        attempt=1,
        error_type="ReportDegraded",
        message=message,
        recovery_tier=RecoveryTier.DEGRADED,
    )
    degraded = ReportResult(
        task_id=task_id,
        output_format=OutputFormat.MARKDOWN,
        title="Report Generation Failed",
        full_content=f"# Report Generation Failed\n\n{message}",
        word_count=5,
        elapsed_seconds=round(elapsed, 3),
        recovery_tier=RecoveryTier.DEGRADED,
    )
    return {
        "report_result": degraded,
        "errors": [error_record],
        "cost_log": tracker_entries,
        "current_agent": None,
        "status": PipelineStatus.PARTIAL,
    }
