"""
api/dependencies.py
────────────────────
Shared FastAPI dependency functions.

The compiled LangGraph pipeline is expensive to build — it wires together
all five agent nodes and compiles the StateGraph. We build it once at
startup and inject it via FastAPI's dependency system.

Also provides: settings injection, request ID generation.
"""

from __future__ import annotations

import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from pipeline.core.config import Settings, get_settings
from pipeline.middleware.logger import get_logger

logger = get_logger(__name__)

# ── Graph singleton ────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    """
    Return the compiled LangGraph pipeline.
    Built once on first call, then cached for the lifetime of the process.
    Injected via FastAPI Depends() so it's testable (overridable in tests).
    """
    global _graph
    if _graph is None:
        logger.info("building_pipeline_graph")
        from pipeline.orchestrator.graph import build_graph
        _graph = build_graph()
        logger.info("pipeline_graph_ready")
    return _graph


def reset_graph() -> None:
    """Force rebuild of the graph on next request. Used in tests."""
    global _graph
    _graph = None


# ── Settings dependency ────────────────────────────────────────────────────────


def get_api_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_api_settings)]


# ── Request ID ────────────────────────────────────────────────────────────────


def get_request_id(
    x_request_id: Annotated[str | None, Header()] = None,
) -> str:
    """
    Return the X-Request-ID header if provided, otherwise generate one.
    Attached to every response for distributed tracing.
    """
    return x_request_id or str(uuid.uuid4())


RequestIdDep = Annotated[str, Depends(get_request_id)]
