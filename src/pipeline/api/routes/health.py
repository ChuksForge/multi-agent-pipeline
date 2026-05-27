"""
api/routes/health.py
─────────────────────
Health check endpoint.

GET /health — returns graph readiness and version.
Used by load balancers, k8s liveness probes, and monitoring.
"""

from __future__ import annotations

from fastapi import APIRouter

from pipeline.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service status and graph readiness.",
)
async def health() -> HealthResponse:
    """Lightweight health check — does not invoke the pipeline."""
    from pipeline.api.dependencies import _graph

    graph_ready = _graph is not None
    return HealthResponse(status="ok", graph_ready=graph_ready)
