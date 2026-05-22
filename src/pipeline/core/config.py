"""
core/config.py
──────────────
Centralised settings loaded from .env.
Pydantic-settings provides type coercion, validation, and IDE autocomplete.
Import `settings` anywhere — it's a singleton.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM API Keys ──────────────────────────────────────────────────────────
    anthropic_api_key: SecretStr = Field(..., description="Anthropic API key")
    openai_api_key: SecretStr | None = Field(None, description="OpenAI API key (optional)")

    # ── Model Configuration ───────────────────────────────────────────────────
    orchestrator_model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-sonnet-4-20250514"
    etl_agent_model: str = "claude-haiku-4-5-20251001"
    analysis_agent_model: str = "claude-haiku-4-5-20251001"
    report_agent_model: str = "claude-haiku-4-5-20251001"
    fallback_model: str = "claude-sonnet-4-20250514"

    # ── Pipeline Behaviour ────────────────────────────────────────────────────
    max_retries: int = Field(3, ge=1, le=10)
    retry_backoff_base: float = Field(2.0, gt=0)
    max_agent_timeout: int = Field(120, ge=10, description="Seconds per agent")
    pipeline_max_rows: int = Field(500_000, ge=100)
    sample_ratio_on_oom: float = Field(0.10, gt=0, le=1.0)

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("data/outputs")
    cost_log_path: Path = Path("data/outputs/cost_log.jsonl")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_format: str = Field("json", pattern="^(json|console)$")

    # ── API ───────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = Field(8000, ge=1024, le=65535)

    # ── Derived helpers ───────────────────────────────────────────────────────

    def model_for_agent(self, agent_id: str) -> str:
        """Return the configured model string for a given agent ID."""
        mapping = {
            "planner": self.planner_model,
            "orchestrator": self.orchestrator_model,
            "etl": self.etl_agent_model,
            "analysis": self.analysis_agent_model,
            "report": self.report_agent_model,
        }
        return mapping.get(agent_id, self.fallback_model)

    @field_validator("outputs_dir", "data_dir", mode="before")
    @classmethod
    def ensure_dir_exists(cls, v: Path | str) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance. Cached after first call."""
    return Settings()


# Module-level singleton — import this directly in most cases
settings = get_settings()
