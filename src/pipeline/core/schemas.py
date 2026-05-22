"""
core/schemas.py
───────────────
Pydantic v2 domain models shared across the entire pipeline.
These are the contracts between agents — strict types, no dicts passed around.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enums ────────────────────────────────────────────────────────────────────


class TaskType(str, Enum):
    ETL_ONLY = "etl_only"
    ANALYSIS_ONLY = "analysis_only"
    REPORT_ONLY = "report_only"
    FULL_PIPELINE = "full_pipeline"


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    PDF = "pdf"


class DataSourceType(str, Enum):
    CSV = "csv"
    PARQUET = "parquet"
    JSON = "json"
    POSTGRES = "postgres"
    DUCKDB = "duckdb"
    UNKNOWN = "unknown"


class TaskComplexity(str, Enum):
    LOW = "low"        # Haiku for all sub-agents
    MEDIUM = "medium"  # Haiku for sub-agents, Sonnet for orchestrator
    HIGH = "high"      # Sonnet for all agents


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PARTIAL = "partial"   # Some agents succeeded, some failed/degraded
    COMPLETE = "complete"
    FAILED = "failed"


class RecoveryTier(str, Enum):
    NONE = "none"
    RETRY = "retry"           # Tier 1: same task, backoff
    SIMPLIFIED = "simplified" # Tier 2: reduced scope
    DEGRADED = "degraded"     # Tier 3: skip agent, continue


# ─── Data Source ──────────────────────────────────────────────────────────────


class DataSource(BaseModel):
    """Represents a single data source the pipeline will ingest."""

    source_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    uri: str = Field(..., description="File path, DB URI, or table name")
    source_type: DataSourceType = DataSourceType.UNKNOWN
    table_name: str | None = Field(None, description="Alias to use in DuckDB queries")
    row_limit: int | None = Field(None, description="Cap rows for large sources")
    options: dict[str, Any] = Field(default_factory=dict, description="Source-specific options")

    @field_validator("uri")
    @classmethod
    def uri_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("DataSource URI cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def infer_source_type(self) -> "DataSource":
        """Auto-detect source type from URI if not explicitly set."""
        if self.source_type == DataSourceType.UNKNOWN:
            uri_lower = self.uri.lower()
            if uri_lower.endswith(".csv"):
                self.source_type = DataSourceType.CSV
            elif uri_lower.endswith(".parquet"):
                self.source_type = DataSourceType.PARQUET
            elif uri_lower.endswith(".json") or uri_lower.endswith(".jsonl"):
                self.source_type = DataSourceType.JSON
            elif uri_lower.startswith("postgresql://") or uri_lower.startswith("postgres://"):
                self.source_type = DataSourceType.POSTGRES
            elif uri_lower.endswith(".duckdb"):
                self.source_type = DataSourceType.DUCKDB
        return self


# ─── SubTask ──────────────────────────────────────────────────────────────────


class SubTask(BaseModel):
    """A single unit of work within a TaskPlan. Has a dependency chain."""

    subtask_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent: str = Field(..., description="Target agent: etl | analysis | report")
    description: str = Field(..., description="Natural language description of this subtask")
    depends_on: list[str] = Field(
        default_factory=list,
        description="subtask_ids that must complete before this one runs",
    )
    required: bool = Field(True, description="If False, pipeline continues on failure")
    estimated_tokens: int = Field(1000, description="Rough token estimate for cost pre-calculation")


# ─── TaskPlan ─────────────────────────────────────────────────────────────────


class TaskPlan(BaseModel):
    """
    Structured output from the Task Planner.
    This is the contract between the Planner and the Orchestrator.
    All routing decisions are derived from this object.
    """

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_task: str = Field(..., description="Original user input, preserved verbatim")
    data_sources: list[DataSource] = Field(..., min_length=0)
    task_type: TaskType
    subtasks: list[SubTask] = Field(..., min_length=1)
    output_format: OutputFormat = OutputFormat.MARKDOWN
    complexity: TaskComplexity = TaskComplexity.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subtasks")
    @classmethod
    def validate_dependency_references(cls, subtasks: list[SubTask]) -> list[SubTask]:
        """All depends_on IDs must reference real subtask_ids in this plan."""
        known_ids = {st.subtask_id for st in subtasks}
        for st in subtasks:
            for dep in st.depends_on:
                if dep not in known_ids:
                    raise ValueError(
                        f"SubTask '{st.subtask_id}' depends on unknown id '{dep}'"
                    )
        return subtasks

    def subtasks_for_agent(self, agent: str) -> list[SubTask]:
        return [st for st in self.subtasks if st.agent == agent]

    def estimated_total_tokens(self) -> int:
        return sum(st.estimated_tokens for st in self.subtasks)


# ─── ETL Result ───────────────────────────────────────────────────────────────


class ColumnSchema(BaseModel):
    """Metadata for a single column after schema inference."""

    name: str
    dtype: str
    nullable: bool
    null_rate: float = Field(ge=0.0, le=1.0)
    unique_count: int | None = None
    sample_values: list[Any] = Field(default_factory=list, max_length=5)


class ValidationIssue(BaseModel):
    """A single data quality problem found during ETL validation."""

    column: str | None = None
    severity: str = Field(..., pattern="^(warning|error)$")
    message: str
    affected_rows: int | None = None


class ETLResult(BaseModel):
    """
    Output contract from the ETL Agent.
    Contains the cleaned dataset reference, inferred schema, and any
    validation issues. Passed directly to the Analysis Agent.
    """

    task_id: str
    source_ids: list[str]
    row_count: int
    column_count: int
    schema: list[ColumnSchema]
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    # Serialised DuckDB view name or in-memory table ref
    duckdb_table_ref: str | None = None
    # Polars DataFrame serialised to JSON for inter-agent passing
    data_json: str | None = Field(None, description="polars JSON serialisation of cleaned data")
    elapsed_seconds: float = 0.0
    recovery_tier: RecoveryTier = RecoveryTier.NONE
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(v.severity == "error" for v in self.validation_issues)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == "warning" for v in self.validation_issues)


# ─── Analysis Result ──────────────────────────────────────────────────────────


class SummaryStats(BaseModel):
    """Descriptive statistics for a single column."""

    column: str
    dtype: str
    count: int
    null_count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    top_values: list[Any] = Field(default_factory=list, max_length=5)


class AnomalyRecord(BaseModel):
    """A single detected anomaly with its row index, value, and detection metadata."""

    row_index: int
    column: str | None = None
    value: Any
    anomaly_score: float
    method: str = Field(..., description="isolation_forest | zscore | ensemble")
    is_anomaly: bool = True


class ChartSpec(BaseModel):
    """Vega-Lite v5 specification. Decoupled from rendering."""

    chart_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str
    spec: dict[str, Any] = Field(..., description="Full Vega-Lite JSON spec")
    description: str = ""


class AnalysisResult(BaseModel):
    """
    Output contract from the Analysis Agent.
    Contains summary statistics, detected anomalies, Vega-Lite chart specs,
    and key findings. Passed directly to the Report Agent.
    """

    task_id: str
    summary_stats: list[SummaryStats] = Field(default_factory=list)
    anomalies: list[AnomalyRecord] = Field(default_factory=list)
    anomaly_rate: float = Field(0.0, ge=0.0, le=1.0)
    charts: list[ChartSpec] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    recovery_tier: RecoveryTier = RecoveryTier.NONE
    warnings: list[str] = Field(default_factory=list)

    @property
    def anomaly_count(self) -> int:
        return len(self.anomalies)


# ─── Report Result ────────────────────────────────────────────────────────────


class ReportSection(BaseModel):
    """A single named section within a rendered report."""

    title: str
    content: str
    order: int


class ReportResult(BaseModel):
    """
    Output contract from the Report Agent.
    Contains the fully rendered report in the requested output format,
    plus section-level breakdown for downstream processing.
    """

    task_id: str
    output_format: OutputFormat
    title: str
    sections: list[ReportSection] = Field(default_factory=list)
    full_content: str = Field(..., description="Complete rendered output (MD / JSON string / PDF path)")
    output_path: str | None = None
    word_count: int = 0
    elapsed_seconds: float = 0.0
    recovery_tier: RecoveryTier = RecoveryTier.NONE


# ─── Agent Error Record ───────────────────────────────────────────────────────


class AgentErrorRecord(BaseModel):
    """Immutable record of a failed agent attempt. Stored in PipelineState.errors."""

    agent_id: str
    attempt: int
    error_type: str
    message: str
    recovery_tier: RecoveryTier
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: dict[str, Any] = Field(default_factory=dict)


# ─── Cost Entry ───────────────────────────────────────────────────────────────


class CostEntry(BaseModel):
    """Single LLM call cost record. Written by TokenTracker middleware."""

    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str
    agent_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    call_type: str = Field("completion", description="completion | tool_call | retry")


class CostSummary(BaseModel):
    """Aggregated cost for a full pipeline run."""

    task_id: str
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_latency_ms: float
    per_agent: dict[str, float] = Field(default_factory=dict, description="agent_id → cost_usd")
    per_model: dict[str, float] = Field(default_factory=dict, description="model → cost_usd")
    entries: list[CostEntry] = Field(default_factory=list)

    @classmethod
    def from_entries(cls, task_id: str, entries: list[CostEntry]) -> "CostSummary":
        per_agent: dict[str, float] = {}
        per_model: dict[str, float] = {}
        for e in entries:
            per_agent[e.agent_id] = per_agent.get(e.agent_id, 0.0) + e.cost_usd
            per_model[e.model] = per_model.get(e.model, 0.0) + e.cost_usd
        return cls(
            task_id=task_id,
            total_cost_usd=sum(e.cost_usd for e in entries),
            total_input_tokens=sum(e.input_tokens for e in entries),
            total_output_tokens=sum(e.output_tokens for e in entries),
            total_latency_ms=sum(e.latency_ms for e in entries),
            per_agent=per_agent,
            per_model=per_model,
            entries=entries,
        )
