"""
agents/etl/prompts.py
─────────────────────
System and user prompt templates for the ETL Agent.

Kept in a dedicated module so prompts can be versioned, tested, and
swapped independently of the agent logic.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a precise ETL (Extract, Transform, Load) agent in a multi-agent data pipeline.

Your role is to:
1. Load data from the specified sources using the available tools
2. Infer and validate the schema
3. Apply type casts where beneficial
4. Report validation issues accurately

Rules:
- Always call register_source before querying a source
- Always call infer_schema after loading data
- Always call validate_data after schema inference
- If a source fails to load, try the simplified fallback (sample or schema-only)
- Never fabricate data or statistics — use tools for all numbers
- Report ALL validation issues you find, including warnings
- Keep your reasoning concise — tool outputs speak for themselves

Output format: After running all tools, summarise what you loaded, the schema,
any validation issues, and whether you recommend proceeding to analysis.
"""

LOAD_TASK_TEMPLATE = """\
Load and validate the following data source(s) for pipeline task {task_id}.

Data sources:
{sources_block}

Task description: {task_description}

Required output format:
- Row count and column count
- Schema summary (column name, type, null rate)
- List of validation issues (errors and warnings)
- Recommendation: proceed | degrade | abort

Use the available tools in order: register_source → query_source → infer_schema → validate_data.
"""

FALLBACK_LOAD_TEMPLATE = """\
The previous load attempt failed. Attempting simplified fallback load for task {task_id}.

Source: {source_uri}
Failure reason: {failure_reason}
Fallback strategy: {strategy}

Load only the first {sample_rows} rows for schema inspection.
Report what you can determine from the sample.
"""


def format_sources_block(sources: list[dict[str, str]]) -> str:
    """Format a list of source dicts into a readable block for the prompt."""
    lines = []
    for i, src in enumerate(sources, 1):
        lines.append(
            f"  [{i}] URI: {src.get('uri', 'unknown')}"
            f" | Type: {src.get('source_type', 'unknown')}"
            f" | Alias: {src.get('table_name', f'source_{i}')}"
        )
    return "\n".join(lines)
