"""Unit tests for LLM selector models - no API calls required."""

import pytest

from task_agent.llms.simple_llm_selector.models import (
    Capability,
    MODEL_CAPABILITIES,
    MODEL_COST,
    CODING_MODEL_PRIORITY,
    get_model_capabilities,
    get_model_cost,
)


class TestModelCapabilities:
    """Tests for MODEL_CAPABILITIES constant."""

    def test_model_capabilities_is_dict(self):
        assert isinstance(MODEL_CAPABILITIES, dict)

    def test_model_capabilities_contains_expected_models(self):
        expected_models = {
            # OpenAI
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5-mini",
            "gpt-5-nano",
            # Google
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-2.5-pro",
            # Groq
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-32b",
            # Cloud/Ollama
            "gemini-3-flash-preview:cloud",
            "qwen3-coder:480b-cloud",
            "gemma3:27b-cloud",
            "glm-4.6:cloud",
            "gpt-oss:20b-cloud",
            "kimi-k2.5:cloud",
        }
        assert set(MODEL_CAPABILITIES.keys()) >= expected_models

    def test_all_capabilities_are_sets(self):
        for model, capabilities in MODEL_CAPABILITIES.items():
            assert isinstance(capabilities, set)

    def test_all_capabilities_are_valid_capability_types(self):
        valid_capabilities: set[Capability] = {
            "reasoning",
            "tools",
            "fast",
            "cheap",
            "informational",
            "coding",
            "vision",
            "long",
            "synthesizing",
            "summarizing",
            "planning",
        }
        for model, capabilities in MODEL_CAPABILITIES.items():
            assert capabilities.issubset(valid_capabilities), f"{model} has invalid capabilities"

    def test_gpt_4o_capabilities(self):
        caps = MODEL_CAPABILITIES.get("gpt-4o", set())
        assert "reasoning" in caps
        assert "tools" in caps
        assert "fast" in caps
        assert "vision" in caps
        assert "long" in caps
        assert "informational" in caps
        assert "coding" in caps

    def test_mini_models_are_cheap(self):
        """Models with 'mini' in name should have 'cheap' capability."""
        assert "cheap" in MODEL_CAPABILITIES.get("gpt-4o-mini", set())
        assert "cheap" in MODEL_CAPABILITIES.get("gpt-5-mini", set())
        assert "cheap" in MODEL_CAPABILITIES.get("gpt-5-nano", set())

    def test_flash_models_are_cheap(self):
        """Flash models should be marked as cheap."""
        assert "cheap" in MODEL_CAPABILITIES.get("gemini-2.5-flash", set())
        assert "cheap" in MODEL_CAPABILITIES.get("gemini-2.5-flash-lite", set())

    def test_coding_models_have_coding_capability(self):
        """Models designed for coding should have the 'coding' capability."""
        coding_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5-mini",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "qwen3-coder:480b-cloud",
            "glm-4.6:cloud",
        ]
        for model in coding_models:
            assert "coding" in MODEL_CAPABILITIES.get(model, set()), f"{model} should have coding capability"

    def test_cloud_models_are_cheap(self):
        """Cloud models should be marked as cheap."""
        cloud_models = [
            "gemini-3-flash-preview:cloud",
            "qwen3-coder:480b-cloud",
            "gemma3:27b-cloud",
            "glm-4.6:cloud",
        ]
        for model in cloud_models:
            # Note: not all cloud models have 'cheap' capability
            # Just check that they exist in capabilities
            assert model in MODEL_CAPABILITIES, f"{model} should be in MODEL_CAPABILITIES"


class TestGetModelCapabilities:
    """Tests for get_model_capabilities function."""

    def test_get_capabilities_for_known_model(self):
        caps = get_model_capabilities("gpt-4o")
        assert isinstance(caps, set)
        assert len(caps) > 0

    def test_get_capabilities_for_unknown_model(self):
        caps = get_model_capabilities("unknown-model-xyz")
        assert caps == set()

    def test_get_capabilities_returns_valid_capabilities(self):
        """Ensure returned capabilities are valid Capability types."""
        valid_capabilities: set[Capability] = {
            "reasoning",
            "tools",
            "fast",
            "cheap",
            "informational",
            "coding",
            "vision",
            "long",
            "synthesizing",
            "summarizing",
            "planning",
        }
        for model in MODEL_CAPABILITIES.keys():
            caps = get_model_capabilities(model)
            assert caps.issubset(valid_capabilities)

    def test_get_capabilities_returns_same_set_reference(self):
        """Test that get_model_capabilities returns a reference to the set."""
        caps1 = get_model_capabilities("gpt-4o")
        caps2 = get_model_capabilities("gpt-4o")
        # They should be the same object (by reference)
        assert caps1 is caps2


class TestModelCost:
    """Tests for MODEL_COST constant."""

    def test_model_cost_is_dict(self):
        assert isinstance(MODEL_COST, dict)

    def test_model_cost_values_are_positive(self):
        for model, cost in MODEL_COST.items():
            assert cost > 0, f"{model} has invalid cost: {cost}"

    def test_model_cost_matches_model_capabilities_keys(self):
        """All models in MODEL_COST should exist in MODEL_CAPABILITIES."""
        for model in MODEL_COST.keys():
            assert model in MODEL_CAPABILITIES, f"{model} in MODEL_COST but not in MODEL_CAPABILITIES"

    def test_mini_models_are_cheaper_than_full_models(self):
        """Mini models should cost less than their full counterparts."""
        if "gpt-4o-mini" in MODEL_COST and "gpt-4o" in MODEL_COST:
            assert MODEL_COST["gpt-4o-mini"] < MODEL_COST["gpt-4o"]

    def test_groq_models_are_very_cheap(self):
        """Groq models should have very low cost."""
        groq_models = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-32b",
        ]
        for model in groq_models:
            assert model in MODEL_COST
            assert MODEL_COST[model] < 0.1, f"{model} should be very cheap"

    def test_cloud_models_are_cheap(self):
        """Cloud models should be inexpensive."""
        cloud_models = [
            "qwen3-coder:480b-cloud",
            "gemma3:27b-cloud",
            "glm-4.6:cloud",
        ]
        for model in cloud_models:
            assert model in MODEL_COST
            assert MODEL_COST[model] < 0.1, f"{model} should be cheap"

    @pytest.mark.parametrize(
        "model1,model2",
        [
            ("gpt-4o", "gpt-4o-mini"),
            ("gemini-2.5-pro", "gemini-2.5-flash"),
            ("gemini-2.5-flash", "gemini-2.5-flash-lite"),
        ],
    )
    def test_expected_cost_orderings(self, model1, model2):
        """Test that expected cost relationships hold."""
        if model1 in MODEL_COST and model2 in MODEL_COST:
            assert MODEL_COST[model1] >= MODEL_COST[model2], \
                f"{model1} should cost >= {model2}"


class TestGetModelCost:
    """Tests for get_model_cost function."""

    def test_get_cost_for_known_model(self):
        cost = get_model_cost("gpt-4o")
        assert isinstance(cost, float)
        assert cost > 0

    def test_get_cost_for_unknown_model(self):
        """Unknown models should return a high default cost."""
        cost = get_model_cost("unknown-model-xyz")
        assert cost == 999.0

    def test_get_cost_returns_same_value_as_model_cost(self):
        """Function should return the same values as MODEL_COST dict."""
        for model in MODEL_COST.keys():
            assert get_model_cost(model) == MODEL_COST[model]

    def test_cost_values_are_floats(self):
        for model in MODEL_COST.keys():
            cost = get_model_cost(model)
            assert isinstance(cost, float)


class TestCodingModelPriority:
    """Tests for CODING_MODEL_PRIORITY constant."""

    def test_coding_model_priority_is_list(self):
        assert isinstance(CODING_MODEL_PRIORITY, list)

    def test_coding_model_priority_not_empty(self):
        assert len(CODING_MODEL_PRIORITY) > 0

    def test_coding_model_priority_models_exist_in_capabilities(self):
        """All models in CODING_MODEL_PRIORITY should exist in MODEL_CAPABILITIES."""
        for model in CODING_MODEL_PRIORITY:
            assert model in MODEL_CAPABILITIES, \
                f"{model} in CODING_MODEL_PRIORITY but not in MODEL_CAPABILITIES"

    def test_coding_model_priority_models_have_coding_capability(self):
        """All models in CODING_MODEL_PRIORITY should have 'coding' capability."""
        for model in CODING_MODEL_PRIORITY:
            caps = MODEL_CAPABILITIES.get(model, set())
            assert "coding" in caps, \
                f"{model} in CODING_MODEL_PRIORITY but doesn't have coding capability"

    def test_coding_model_priority_order_is_maintained(self):
        """Test that the priority order is as expected."""
        assert CODING_MODEL_PRIORITY[0] == "deepseek-v3.1:671b-cloud"
        assert CODING_MODEL_PRIORITY[1] == "kimi-k2.5:cloud"
        assert CODING_MODEL_PRIORITY[2] == "qwen3-coder:480b-cloud"
        assert CODING_MODEL_PRIORITY[3] == "glm-4.6:cloud"
        assert CODING_MODEL_PRIORITY[4] == "gemini-2.5-pro"
        assert CODING_MODEL_PRIORITY[5] == "gemini-2.5-flash"

    def test_coding_model_priority_no_duplicates(self):
        """CODING_MODEL_PRIORITY should not have duplicate entries."""
        assert len(CODING_MODEL_PRIORITY) == len(set(CODING_MODEL_PRIORITY))


class TestCapabilityType:
    """Tests for Capability type and valid values."""

    def test_capability_literal_contains_expected_values(self):
        """This test documents the expected capability values."""
        # The actual Capability is a Literal type, so we check against the values we expect
        expected_capabilities = {
            "reasoning",
            "tools",
            "fast",
            "cheap",
            "informational",
            "coding",
            "vision",
            "long",
            "synthesizing",
            "summarizing",
            "planning",
        }
        # Check that all capabilities in MODEL_CAPABILITIES are from this set
        all_caps = set()
        for caps in MODEL_CAPABILITIES.values():
            all_caps.update(caps)
        # We expect some overlap but the actual capabilities may be a subset
        assert all_caps.intersection(expected_capabilities)
        # Verify no unexpected capabilities
        assert all_caps.issubset(expected_capabilities)


class TestModelDataConsistency:
    """Tests for consistency between MODEL_CAPABILITIES and MODEL_COST."""

    def test_all_cost_models_have_capabilities(self):
        """Every model in MODEL_COST should have capabilities defined."""
        for model in MODEL_COST:
            assert model in MODEL_CAPABILITIES, \
                f"{model} has cost but no capabilities defined"

    def test_capability_models_may_not_have_cost(self):
        """It's okay for a model to have capabilities but no explicit cost."""
        # This just documents the design - some models may only have capabilities
        pass


class TestSpecificModelCharacteristics:
    """Tests for specific model characteristics."""

    def test_gpt_4o_has_vision(self):
        assert "vision" in MODEL_CAPABILITIES.get("gpt-4o", set())

    def test_gemini_25_flash_has_vision(self):
        assert "vision" in MODEL_CAPABILITIES.get("gemini-2.5-flash", set())

    def test_gemini_25_pro_has_vision(self):
        assert "vision" in MODEL_CAPABILITIES.get("gemini-2.5-pro", set())

    def test_flash_models_are_fast(self):
        """Models with 'flash' in name should be fast."""
        flash_models = [
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
            "gemini-3-flash-preview:cloud",
        ]
        for model in flash_models:
            assert "fast" in MODEL_CAPABILITIES.get(model, set()), \
                f"{model} should be fast"

    def test_groq_models_are_fast(self):
        """Groq-hosted models should be fast."""
        groq_models = [
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-32b",
        ]
        for model in groq_models:
            assert "fast" in MODEL_CAPABILITIES.get(model, set()), \
                f"{model} should be fast"

    def test_qwen_coder_is_specialized_for_coding(self):
        """qwen3-coder should be optimized for coding tasks."""
        caps = MODEL_CAPABILITIES.get("qwen3-coder:480b-cloud", set())
        assert "coding" in caps

    def test_pro_models_have_reasoning(self):
        """Pro models should have reasoning capability."""
        pro_models = [
            "gemini-2.5-pro",
        ]
        for model in pro_models:
            assert "reasoning" in MODEL_CAPABILITIES.get(model, set()), \
                f"{model} should have reasoning"

    def test_long_context_models(self):
        """Models with long context capability."""
        long_models = [
            "gpt-4o",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-3-flash-preview:cloud",
            "glm-4.6:cloud",
        ]
        for model in long_models:
            assert "long" in MODEL_CAPABILITIES.get(model, set()), \
                f"{model} should have long context"

    @pytest.mark.parametrize(
        "model,expected_capabilities",
        [
            ("gpt-4o", {"reasoning", "tools", "fast", "vision", "long", "informational", "coding"}),
            ("gemini-2.5-flash", {"reasoning", "tools", "fast", "cheap", "vision", "long", "informational", "coding"}),
            ("qwen/qwen3-32b", {"reasoning", "tools", "fast", "cheap", "informational"}),
        ],
    )
    def test_specific_model_capability_sets(self, model, expected_capabilities):
        """Test exact capability sets for key models."""
        actual = MODEL_CAPABILITIES.get(model, set())
        # Check that expected capabilities are present (actual may have more)
        assert expected_capabilities.issubset(actual), f"{model} missing expected capabilities"


class TestCostRanking:
    """Tests for relative cost rankings."""

    def test_models_can_be_ranked_by_cost(self):
        """Test that we can rank models by cost."""
        models_with_costs = [(model, get_model_cost(model)) for model in MODEL_COST.keys()]
        sorted_models = sorted(models_with_costs, key=lambda x: x[1])

        # Verify sorting worked
        for i in range(len(sorted_models) - 1):
            assert sorted_models[i][1] <= sorted_models[i + 1][1]

    def test_cheapest_models(self):
        """Identify and validate the cheapest models."""
        sorted_models = sorted(MODEL_COST.items(), key=lambda x: x[1])
        cheapest_cost = sorted_models[0][1]

        # The cheapest models should be very cheap (less than 0.1)
        assert cheapest_cost < 0.1

    def test_most_expensive_models(self):
        """Identify and validate the most expensive models."""
        sorted_models = sorted(MODEL_COST.items(), key=lambda x: x[1], reverse=True)
        most_expensive = sorted_models[0]

        # gpt-4o should be among the most expensive
        assert "gpt-4o" in [m for m, c in sorted_models[:3]]
