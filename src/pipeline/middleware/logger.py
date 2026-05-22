"""
middleware/logger.py
────────────────────
Structured logging setup using structlog.

Every log event is a JSON object with:
  - timestamp
  - level
  - logger name
  - event (message)
  - arbitrary key=value context

Console output gets colourised rendering in development.
JSON output (production / CI) is machine-parseable.

Usage:
    from pipeline.middleware.logger import get_logger
    logger = get_logger(__name__)
    logger.info("etl_complete", rows=50000, elapsed_ms=1234.5)
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from typing import Any

import structlog

_configured = False


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Call once at application startup (main.py / run_demo.py).
    Idempotent — safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if fmt == "console":
        renderer: Any = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quiet noisy third-party libs
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


@lru_cache(maxsize=None)
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger. Cached per name."""
    return structlog.get_logger(name)


# ─── Context helpers ──────────────────────────────────────────────────────────


def bind_pipeline_context(task_id: str, agent_id: str | None = None) -> None:
    """
    Bind task_id (and optionally agent_id) to the current context.
    All subsequent log calls in this context will include these fields.
    Call at the start of each agent node.
    """
    ctx: dict[str, str] = {"task_id": task_id}
    if agent_id:
        ctx["agent_id"] = agent_id
    structlog.contextvars.bind_contextvars(**ctx)


def clear_pipeline_context() -> None:
    """Clear bound context at the end of a pipeline run."""
    structlog.contextvars.clear_contextvars()
