"""
api/main.py
────────────
FastAPI application factory.

Usage:
    uvicorn pipeline.api.main:app --reload --port 8000

    # or via script:
    python -m pipeline.api.main
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pipeline.api.dependencies import get_graph
from pipeline.api.routes import health, pipeline
from pipeline.middleware.logger import configure_logging, get_logger

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Builds the pipeline graph at startup so the first request isn't slow.
    """
    configure_logging(level="INFO", fmt="json")
    logger.info("api_startup", version="0.1.0")

    # Pre-build the graph so it's ready for the first request
    try:
        get_graph()
        logger.info("graph_preloaded")
    except Exception as e:
        logger.warning("graph_preload_failed", error=str(e),
                       note="Graph will build on first request")

    yield

    logger.info("api_shutdown")


# ── App factory ───────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Call this in tests to get a fresh app instance.
    """
    app = FastAPI(
        title="Multi-Agent Data Pipeline",
        description=(
            "Production-grade multi-agent pipeline for ETL, anomaly detection, "
            "and report generation. Powered by LangGraph and Claude."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing middleware ─────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next) -> Response:
        start = time.monotonic()
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        response.headers["X-Process-Time-Ms"] = str(round(elapsed_ms, 1))
        return response

    # ── Global exception handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": str(exc)},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(pipeline.router, prefix="/api/v1")

    # ── Root redirect ─────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        return {"message": "Multi-Agent Data Pipeline API", "docs": "/docs"}

    return app


# ── Module-level app instance ─────────────────────────────────────────────────
# uvicorn pipeline.api.main:app

app = create_app()


if __name__ == "__main__":
    import uvicorn
    from pipeline.core.config import settings

    uvicorn.run(
        "pipeline.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
