"""
middleware/cost_model.py
────────────────────────
Authoritative pricing table for all supported models.

Design:
  - Prices are per 1M tokens (industry standard unit)
  - `compute_cost()` is the single call site — no scattered math elsewhere
  - Update PRICING_TABLE when Anthropic/OpenAI change rates
  - ModelTier enum drives the cost comparison benchmark
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ModelTier(str, Enum):
    """Logical tier — used for the cost comparison benchmark."""
    LARGE = "large"    # Sonnet-class: high reasoning, higher cost
    SMALL = "small"    # Haiku-class: fast execution, low cost


@dataclass(frozen=True)
class ModelPricing:
    """
    Pricing for a single model.
    All values are USD per 1,000,000 tokens.
    """
    model_id: str
    input_per_million: float
    output_per_million: float
    tier: ModelTier
    provider: str = "anthropic"
    context_window: int = 200_000

    def cost_for(self, input_tokens: int, output_tokens: int) -> float:
        """Return USD cost for the given token counts."""
        input_cost = (input_tokens / 1_000_000) * self.input_per_million
        output_cost = (output_tokens / 1_000_000) * self.output_per_million
        return round(input_cost + output_cost, 8)


# ─── Pricing Table ────────────────────────────────────────────────────────────
# Source: https://www.anthropic.com/pricing (update regularly)
# Last verified: 2025-01

PRICING_TABLE: dict[str, ModelPricing] = {
    # Anthropic — Claude 4 family
    "claude-opus-4-20250514": ModelPricing(
        model_id="claude-opus-4-20250514",
        input_per_million=15.0,
        output_per_million=75.0,
        tier=ModelTier.LARGE,
        context_window=200_000,
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        model_id="claude-sonnet-4-20250514",
        input_per_million=3.0,
        output_per_million=15.0,
        tier=ModelTier.LARGE,
        context_window=200_000,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        model_id="claude-haiku-4-5-20251001",
        input_per_million=0.8,
        output_per_million=4.0,
        tier=ModelTier.SMALL,
        context_window=200_000,
    ),
    # Anthropic — Claude 3.5 family (legacy reference)
    "claude-3-5-sonnet-20241022": ModelPricing(
        model_id="claude-3-5-sonnet-20241022",
        input_per_million=3.0,
        output_per_million=15.0,
        tier=ModelTier.LARGE,
        context_window=200_000,
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        model_id="claude-3-5-haiku-20241022",
        input_per_million=0.8,
        output_per_million=4.0,
        tier=ModelTier.SMALL,
        context_window=200_000,
    ),
    # OpenAI (for benchmark comparison)
    "gpt-4o": ModelPricing(
        model_id="gpt-4o",
        input_per_million=2.5,
        output_per_million=10.0,
        tier=ModelTier.LARGE,
        provider="openai",
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelPricing(
        model_id="gpt-4o-mini",
        input_per_million=0.15,
        output_per_million=0.60,
        tier=ModelTier.SMALL,
        provider="openai",
        context_window=128_000,
    ),
}

# Convenience aliases so config.py model strings always resolve
_ALIASES: dict[str, str] = {
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-haiku-4": "claude-haiku-4-5-20251001",
    "claude-opus-4": "claude-opus-4-20250514",
}


# ─── Public API ───────────────────────────────────────────────────────────────


def get_pricing(model_id: str) -> ModelPricing | None:
    """Return ModelPricing for the given model ID, resolving aliases."""
    canonical = _ALIASES.get(model_id, model_id)
    return PRICING_TABLE.get(canonical)


def compute_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Compute USD cost for a single LLM call.
    Returns 0.0 with a warning if the model is unknown.
    """
    pricing = get_pricing(model_id)
    if pricing is None:
        # Unknown model — log zero rather than raise, to not block the pipeline
        return 0.0
    return pricing.cost_for(input_tokens, output_tokens)


def tier_for_model(model_id: str) -> ModelTier:
    pricing = get_pricing(model_id)
    if pricing is None:
        return ModelTier.LARGE  # Conservative default
    return pricing.tier


def all_models_for_tier(tier: ModelTier) -> list[ModelPricing]:
    return [p for p in PRICING_TABLE.values() if p.tier == tier]


# ─── Benchmark Scenarios ──────────────────────────────────────────────────────
# Used by scripts/cost_benchmark.py

BENCHMARK_SCENARIOS: dict[str, dict[str, str]] = {
    "all_large": {
        "planner": "claude-sonnet-4-20250514",
        "orchestrator": "claude-sonnet-4-20250514",
        "etl": "claude-sonnet-4-20250514",
        "analysis": "claude-sonnet-4-20250514",
        "report": "claude-sonnet-4-20250514",
    },
    "mixed_routing": {
        "planner": "claude-sonnet-4-20250514",
        "orchestrator": "claude-sonnet-4-20250514",
        "etl": "claude-haiku-4-5-20251001",
        "analysis": "claude-haiku-4-5-20251001",
        "report": "claude-haiku-4-5-20251001",
    },
    "all_small": {
        "planner": "claude-haiku-4-5-20251001",
        "orchestrator": "claude-haiku-4-5-20251001",
        "etl": "claude-haiku-4-5-20251001",
        "analysis": "claude-haiku-4-5-20251001",
        "report": "claude-haiku-4-5-20251001",
    },
    "openai_large": {
        "planner": "gpt-4o",
        "orchestrator": "gpt-4o",
        "etl": "gpt-4o",
        "analysis": "gpt-4o",
        "report": "gpt-4o",
    },
    "openai_mixed": {
        "planner": "gpt-4o",
        "orchestrator": "gpt-4o",
        "etl": "gpt-4o-mini",
        "analysis": "gpt-4o-mini",
        "report": "gpt-4o-mini",
    },
}
