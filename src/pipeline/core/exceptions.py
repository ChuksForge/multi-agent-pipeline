"""
core/exceptions.py
──────────────────
All pipeline-specific exceptions. Typed and granular so the orchestrator
can make precise recovery decisions — not just catch Exception everywhere.
"""

from __future__ import annotations

from typing import Any


# ─── Base ─────────────────────────────────────────────────────────────────────


class PipelineError(Exception):
    """Root exception for all pipeline errors. Carry structured context."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.args[0]!r}, context={self.context})"


# ─── Agent Errors ─────────────────────────────────────────────────────────────


class AgentError(PipelineError):
    """Raised when an agent fails at the node level."""

    def __init__(
        self,
        agent_id: str,
        message: str,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, context)
        self.agent_id = agent_id
        self.recoverable = recoverable


class AgentTimeoutError(AgentError):
    """Agent exceeded its allotted execution time."""

    def __init__(self, agent_id: str, timeout_seconds: float) -> None:
        super().__init__(
            agent_id=agent_id,
            message=f"Agent '{agent_id}' timed out after {timeout_seconds}s",
            recoverable=True,
            context={"timeout_seconds": timeout_seconds},
        )


class AgentRetryExhaustedError(AgentError):
    """All retry attempts for an agent have been exhausted."""

    def __init__(self, agent_id: str, attempts: int) -> None:
        super().__init__(
            agent_id=agent_id,
            message=f"Agent '{agent_id}' failed after {attempts} retry attempts",
            recoverable=False,
            context={"attempts": attempts},
        )


# ─── ETL Errors ───────────────────────────────────────────────────────────────


class ETLError(PipelineError):
    """Base for ETL-layer errors."""


class DataSourceNotFoundError(ETLError):
    """The specified data source path or URI does not exist."""

    def __init__(self, source: str) -> None:
        super().__init__(
            f"Data source not found: '{source}'",
            context={"source": source},
        )


class SchemaInferenceError(ETLError):
    """DuckDB / polars could not infer a usable schema."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(
            f"Schema inference failed for '{source}': {detail}",
            context={"source": source, "detail": detail},
        )


class DataValidationError(ETLError):
    """
    Data failed validation checks.
    ``failures`` is a list of human-readable descriptions.
    """

    def __init__(self, failures: list[str]) -> None:
        joined = "; ".join(failures)
        super().__init__(
            f"Data validation failed: {joined}",
            context={"failures": failures},
        )


class UnsupportedFileFormatError(ETLError):
    """File extension is not supported by the ETL agent."""

    def __init__(self, extension: str) -> None:
        super().__init__(
            f"Unsupported file format: '{extension}'",
            context={"extension": extension},
        )


# ─── Analysis Errors ──────────────────────────────────────────────────────────


class AnalysisError(PipelineError):
    """Base for analysis-layer errors."""


class InsufficientDataError(AnalysisError):
    """Not enough rows/columns to perform the requested analysis."""

    def __init__(self, required: int, actual: int) -> None:
        super().__init__(
            f"Insufficient data: need {required} rows, got {actual}",
            context={"required": required, "actual": actual},
        )


class AnomalyDetectionError(AnalysisError):
    """Anomaly detection failed (OOM, convergence, etc.)."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"Anomaly detection error: {detail}",
            context={"detail": detail},
        )


# ─── Report Errors ────────────────────────────────────────────────────────────


class ReportError(PipelineError):
    """Base for report-layer errors."""


class OutputFormatError(ReportError):
    """Requested output format is not supported or cannot be rendered."""

    def __init__(self, fmt: str) -> None:
        super().__init__(
            f"Output format not supported: '{fmt}'",
            context={"format": fmt},
        )


# ─── Orchestration Errors ─────────────────────────────────────────────────────


class OrchestratorError(PipelineError):
    """Errors from the LangGraph supervisor layer."""


class TaskPlanningError(OrchestratorError):
    """The planner LLM failed to produce a valid TaskPlan."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"Task planning failed: {detail}",
            context={"detail": detail},
        )


class InvalidTaskError(OrchestratorError):
    """The incoming task cannot be interpreted or is malformed."""

    def __init__(self, raw_task: str) -> None:
        super().__init__(
            f"Invalid or unrecognisable task: '{raw_task[:200]}'",
            context={"raw_task": raw_task},
        )


# ─── Cost / Middleware Errors ─────────────────────────────────────────────────


class CostTrackingError(PipelineError):
    """Token usage metadata was missing or malformed."""
