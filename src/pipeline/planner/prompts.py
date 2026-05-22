"""
planner/prompts.py
──────────────────
Prompts for the Task Planner.

The planner makes one LLM call that produces a structured TaskPlan.
The system prompt instructs the model to output valid JSON matching
the TaskPlan schema — parsed and validated by Pydantic v2.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a data pipeline task planner. Given a natural language data task,
you decompose it into a structured execution plan for a multi-agent pipeline.

The pipeline has three agents:
  - etl:      loads, validates, and cleans data from files or databases
  - analysis: computes statistics and detects anomalies
  - report:   generates a human-readable report

Rules:
  - Always include at least one subtask
  - Subtasks must list their dependencies by subtask_id
  - Only use agents: "etl", "analysis", "report"
  - task_type must be one of: etl_only, analysis_only, report_only, full_pipeline
  - output_format must be one of: markdown, json, pdf
  - complexity must be one of: low, medium, high
  - data_sources is a list of URIs or table names mentioned in the task
  - If no data source is mentioned, use an empty list
  - estimated_tokens is your rough estimate of LLM tokens per subtask (100-2000)

Respond with ONLY valid JSON — no preamble, no explanation, no markdown fences.
The JSON must match this exact schema:

{
  "raw_task": "<the original task string>",
  "data_sources": [
    {"uri": "<path or URI>", "table_name": "<optional alias>"}
  ],
  "task_type": "<etl_only|analysis_only|report_only|full_pipeline>",
  "output_format": "<markdown|json|pdf>",
  "complexity": "<low|medium|high>",
  "subtasks": [
    {
      "subtask_id": "<short unique id like st-001>",
      "agent": "<etl|analysis|report>",
      "description": "<what this subtask does>",
      "depends_on": ["<subtask_id>"],
      "required": true,
      "estimated_tokens": 500
    }
  ]
}
"""

PLAN_TASK_TEMPLATE = """\
Plan the following data pipeline task:

Task: {raw_task}

Additional context:
- Available data sources: {sources_hint}
- Preferred output format: {output_format_hint}
- Current date: {current_date}

Produce a complete TaskPlan JSON object.
"""


def format_plan_prompt(
    raw_task: str,
    sources_hint: str = "not specified",
    output_format_hint: str = "markdown",
    current_date: str = "",
) -> str:
    return PLAN_TASK_TEMPLATE.format(
        raw_task=raw_task,
        sources_hint=sources_hint,
        output_format_hint=output_format_hint,
        current_date=current_date or "not specified",
    )
