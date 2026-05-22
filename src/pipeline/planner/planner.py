"""
planner/planner.py
──────────────────
Task Planner — converts a natural language task string into a typed TaskPlan.

One LLM call (Sonnet) that returns structured JSON.
Validated by Pydantic v2 — if validation fails, raises TaskPlanningError.

Design decisions:
  - Uses Sonnet for planning (needs reliable structured output reasoning)
  - JSON parsed directly from LLM response — no instructor/tool_use dependency
  - Retry logic: up to 2 attempts if JSON parse or Pydantic validation fails
  - Assigns a fresh task_id via uuid4 — never trusts the LLM to generate IDs
  - DataSource.source_type auto-inferred from URI extension (see schemas.py)
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_anthropic import ChatAnthropic

from pipeline.core.config import settings
from pipeline.core.exceptions import InvalidTaskError, TaskPlanningError
from pipeline.core.schemas import (
    DataSource,
    OutputFormat,
    SubTask,
    TaskComplexity,
    TaskPlan,
    TaskType,
)
from pipeline.middleware.logger import get_logger
from pipeline.middleware.token_tracker import make_tracker
from pipeline.planner.prompts import SYSTEM_PROMPT, format_plan_prompt

logger = get_logger(__name__)

_MAX_PLANNING_ATTEMPTS = 2


def plan_task(
    raw_task: str,
    task_id: str | None = None,
    sources_hint: str = "not specified",
    output_format_hint: str = "markdown",
) -> TaskPlan:
    """
    Convert a natural language task string into a validated TaskPlan.

    Args:
        raw_task:           User's task description.
        task_id:            Override task_id (generated if None).
        sources_hint:       Hint about available data sources for the prompt.
        output_format_hint: Preferred output format hint for the prompt.

    Returns:
        Validated TaskPlan ready for the orchestrator.

    Raises:
        InvalidTaskError:    raw_task is empty or too short.
        TaskPlanningError:   LLM failed to produce a valid plan after retries.
    """
    raw_task = raw_task.strip()
    if len(raw_task) < 5:
        raise InvalidTaskError(raw_task)

    final_task_id = task_id or str(uuid.uuid4())
    tracker = make_tracker(task_id=final_task_id, agent_id="planner")

    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = format_plan_prompt(
        raw_task=raw_task,
        sources_hint=sources_hint,
        output_format_hint=output_format_hint,
        current_date=current_date,
    )

    llm = ChatAnthropic(
        model=settings.planner_model,
        max_tokens=1024,
        callbacks=[tracker],
    )

    last_error: Exception | None = None

    for attempt in range(1, _MAX_PLANNING_ATTEMPTS + 1):
        try:
            logger.debug("planner_attempt", attempt=attempt, task_id=final_task_id)

            response = llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ])

            content = _extract_content(response)
            plan_dict = _parse_json(content)
            plan = _build_task_plan(plan_dict, raw_task, final_task_id)

            logger.info(
                "planner_complete",
                task_id=final_task_id,
                task_type=plan.task_type.value,
                subtasks=len(plan.subtasks),
                complexity=plan.complexity.value,
                cost_usd=tracker.total_cost_usd,
            )
            return plan

        except (InvalidTaskError, TaskPlanningError):
            raise
        except Exception as e:
            last_error = e
            logger.warning(
                "planner_attempt_failed",
                attempt=attempt,
                error=str(e),
                task_id=final_task_id,
            )

    raise TaskPlanningError(
        f"Failed after {_MAX_PLANNING_ATTEMPTS} attempts. "
        f"Last error: {last_error}"
    )


# ── Parsing helpers ───────────────────────────────────────────────────────────


def _extract_content(response: Any) -> str:
    """Pull text content from a LangChain response object."""
    content = response.content
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
        )
    return str(content).strip()


def _parse_json(content: str) -> dict[str, Any]:
    """
    Parse JSON from LLM response content.
    Strips markdown fences if present, then validates it's a dict.
    """
    # Strip ```json ... ``` fences
    clean = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    clean = re.sub(r"\s*```\s*$", "", clean, flags=re.MULTILINE)
    clean = clean.strip()

    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        raise TaskPlanningError(f"LLM returned invalid JSON: {e}. Content: {clean[:200]}")

    if not isinstance(parsed, dict):
        raise TaskPlanningError(f"Expected JSON object, got {type(parsed).__name__}")

    return parsed


def _build_task_plan(data: dict[str, Any], raw_task: str, task_id: str) -> TaskPlan:
    """
    Build and validate a TaskPlan from the parsed LLM JSON.
    Maps LLM field names to Pydantic models with fallback defaults.
    """
    # Data sources
    sources_raw = data.get("data_sources", [])
    data_sources: list[DataSource] = []
    for src in sources_raw:
        if isinstance(src, str):
            data_sources.append(DataSource(uri=src))
        elif isinstance(src, dict) and src.get("uri"):
            data_sources.append(DataSource(
                uri=src["uri"],
                table_name=src.get("table_name"),
            ))

    # Task type
    try:
        task_type = TaskType(data.get("task_type", "full_pipeline"))
    except ValueError:
        task_type = TaskType.FULL_PIPELINE

    # Output format
    try:
        output_format = OutputFormat(data.get("output_format", "markdown"))
    except ValueError:
        output_format = OutputFormat.MARKDOWN

    # Complexity
    try:
        complexity = TaskComplexity(data.get("complexity", "medium"))
    except ValueError:
        complexity = TaskComplexity.MEDIUM

    # Subtasks
    subtasks_raw = data.get("subtasks", [])
    if not subtasks_raw:
        # Fallback: generate a minimal full-pipeline plan
        subtasks_raw = _default_subtasks(task_type)

    subtasks: list[SubTask] = []
    for st in subtasks_raw:
        if not isinstance(st, dict):
            continue
        subtasks.append(SubTask(
            subtask_id=st.get("subtask_id", f"st-{len(subtasks)+1:03d}"),
            agent=st.get("agent", "etl"),
            description=st.get("description", ""),
            depends_on=st.get("depends_on", []),
            required=st.get("required", True),
            estimated_tokens=int(st.get("estimated_tokens", 500)),
        ))

    if not subtasks:
        raise TaskPlanningError("Planner produced zero valid subtasks")

    try:
        return TaskPlan(
            task_id=task_id,
            raw_task=raw_task,
            data_sources=data_sources,
            task_type=task_type,
            subtasks=subtasks,
            output_format=output_format,
            complexity=complexity,
        )
    except Exception as e:
        raise TaskPlanningError(f"TaskPlan validation failed: {e}")


def _default_subtasks(task_type: TaskType) -> list[dict[str, Any]]:
    """
    Generate a minimal subtask list when the LLM omits them.
    This is the fallback — the planner should always produce subtasks.
    """
    if task_type == TaskType.ETL_ONLY:
        return [{"subtask_id": "st-001", "agent": "etl",
                 "description": "Load and validate data", "depends_on": []}]
    if task_type == TaskType.ANALYSIS_ONLY:
        return [{"subtask_id": "st-001", "agent": "analysis",
                 "description": "Analyse data", "depends_on": []}]
    if task_type == TaskType.REPORT_ONLY:
        return [{"subtask_id": "st-001", "agent": "report",
                 "description": "Generate report", "depends_on": []}]
    # FULL_PIPELINE
    return [
        {"subtask_id": "st-001", "agent": "etl",
         "description": "Load and validate data", "depends_on": []},
        {"subtask_id": "st-002", "agent": "analysis",
         "description": "Analyse and detect anomalies", "depends_on": ["st-001"]},
        {"subtask_id": "st-003", "agent": "report",
         "description": "Generate report", "depends_on": ["st-002"]},
    ]
