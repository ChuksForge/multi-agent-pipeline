"""
orchestrator/prompts.py
────────────────────────
Prompts for the Orchestrator supervisor node.

The supervisor does NOT make LLM calls — routing decisions are purely
deterministic from PipelineState. These prompts are reserved for future
use if an LLM-based routing decision is ever needed.
"""

from __future__ import annotations

# Routing is deterministic — no LLM prompt needed for the supervisor.
# This module is a placeholder for future LLM-assisted routing decisions
# (e.g. deciding whether to retry or degrade based on error context).

SUPERVISOR_CONTEXT = """\
You are the orchestrator of a multi-agent data pipeline.
Agents: etl, analysis, report.
Your job: route tasks, handle retries, and decide when to degrade gracefully.
"""
