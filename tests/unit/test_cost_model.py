"""
tests/unit/test_cost_model.py
─────────────────────────────
Tests for the pricing table and cost calculation utilities.
"""

from __future__ import annotations

import pytest

from pipeline.middleware.cost_model import (
    BENCHMARK_SCENARIOS,
    PRICING_TABLE,
    ModelTier,
    all_models_for_tier,
    compute_cost,
    get_pricing,
    tier_for_model,
)


class TestPricingTable:
    def test_all_required_models_present(self):
        required = [
            "claude-sonnet-4-20250514",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
        ]
        for model in required:
            assert model in PRICING_TABLE, f"Missing model: {model}"

    def test_all_prices_positive(self):
        for model_id, pricing in PRICING_TABLE.items():
            assert pricing.input_per_million > 0, f"{model_id} input price must be > 0"
            assert pricing.output_per_million > 0, f"{model_id} output price must be > 0"

    def test_output_always_more_expensive_than_input(self):
        """Output tokens are always more expensive than input for all models."""
        for model_id, pricing in PRICING_TABLE.items():
            assert pricing.output_per_million >= pricing.input_per_million, (
                f"{model_id}: output price should be >= input price"
            )

    def test_small_models_cheaper_than_large(self):
        small_models = all_models_for_tier(ModelTier.SMALL)
        large_models = all_models_for_tier(ModelTier.LARGE)
        min_large_input = min(m.input_per_million for m in large_models)
        max_small_input = max(m.input_per_million for m in small_models)
        assert max_small_input < min_large_input, (
            "All SMALL models should be cheaper than all LARGE models"
        )


class TestGetPricing:
    def test_known_model_returns_pricing(self):
        p = get_pricing("claude-sonnet-4-20250514")
        assert p is not None
        assert p.model_id == "claude-sonnet-4-20250514"

    def test_alias_resolves(self):
        p = get_pricing("claude-sonnet-4")
        assert p is not None
        assert p.model_id == "claude-sonnet-4-20250514"

    def test_unknown_model_returns_none(self):
        p = get_pricing("gpt-999-ultra")
        assert p is None


class TestComputeCost:
    def test_zero_tokens_returns_zero(self):
        cost = compute_cost("claude-haiku-4-5-20251001", 0, 0)
        assert cost == pytest.approx(0.0)

    def test_known_model_computes_correctly(self):
        # Haiku: $0.80/M input, $4.00/M output
        # 1M input + 500k output → 0.80 + 2.00 = $2.80
        cost = compute_cost("claude-haiku-4-5-20251001", 1_000_000, 500_000)
        assert cost == pytest.approx(2.80)

    def test_sonnet_more_expensive_than_haiku(self):
        tokens_in, tokens_out = 10_000, 5_000
        haiku_cost = compute_cost("claude-haiku-4-5-20251001", tokens_in, tokens_out)
        sonnet_cost = compute_cost("claude-sonnet-4-20250514", tokens_in, tokens_out)
        assert sonnet_cost > haiku_cost

    def test_unknown_model_returns_zero(self):
        cost = compute_cost("nonexistent-model", 1000, 500)
        assert cost == pytest.approx(0.0)

    def test_alias_works_in_compute(self):
        cost_alias = compute_cost("claude-haiku-4", 1000, 500)
        cost_full = compute_cost("claude-haiku-4-5-20251001", 1000, 500)
        assert cost_alias == cost_full

    def test_cost_is_rounded_to_8_decimals(self):
        cost = compute_cost("claude-haiku-4-5-20251001", 100, 50)
        # Result should have at most 8 decimal places
        assert cost == round(cost, 8)


class TestTierForModel:
    def test_sonnet_is_large(self):
        assert tier_for_model("claude-sonnet-4-20250514") == ModelTier.LARGE

    def test_haiku_is_small(self):
        assert tier_for_model("claude-haiku-4-5-20251001") == ModelTier.SMALL

    def test_unknown_model_defaults_to_large(self):
        # Conservative default — don't undercount cost for unknown models
        assert tier_for_model("mystery-model") == ModelTier.LARGE


class TestBenchmarkScenarios:
    def test_all_required_scenarios_present(self):
        required = ["all_large", "mixed_routing", "all_small"]
        for s in required:
            assert s in BENCHMARK_SCENARIOS, f"Missing scenario: {s}"

    def test_all_scenarios_have_five_agents(self):
        required_agents = {"planner", "orchestrator", "etl", "analysis", "report"}
        for name, scenario in BENCHMARK_SCENARIOS.items():
            assert set(scenario.keys()) == required_agents, (
                f"Scenario '{name}' missing agents: {required_agents - set(scenario.keys())}"
            )

    def test_mixed_routing_uses_small_for_execution(self):
        scenario = BENCHMARK_SCENARIOS["mixed_routing"]
        for agent in ("etl", "analysis", "report"):
            tier = tier_for_model(scenario[agent])
            assert tier == ModelTier.SMALL, f"Expected small model for {agent} in mixed_routing"

    def test_mixed_routing_uses_large_for_planning(self):
        scenario = BENCHMARK_SCENARIOS["mixed_routing"]
        for agent in ("planner", "orchestrator"):
            tier = tier_for_model(scenario[agent])
            assert tier == ModelTier.LARGE, f"Expected large model for {agent} in mixed_routing"
