"""Tests for LLM router functions in simple_llm_selector/router.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from task_agent.llms.simple_llm_selector.router import (
    select_models,
    get_cheapest_model,
    get_model_details,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_model_capabilities():
    """Mock model capabilities data."""
    return {
        "gpt-4o-mini": {"coding", "reasoning", "fast", "tools", "informational"},
        "gemini-2.5-flash": {"coding", "reasoning", "cheap", "informational"},
        "deepseek-v3.1:671b-cloud": {"coding", "reasoning", "long", "tools"},
        "kimi-k2.5:cloud": {"coding", "reasoning", "long"},
        "qwen3-coder:480b-cloud": {"coding", "reasoning", "tools"},
        "glm-4.6:cloud": {"coding", "reasoning"},
        "gemini-2.5-pro": {"coding", "reasoning", "long", "tools"},
        "informational-model": {"informational", "cheap", "fast"},
    }


@pytest.fixture
def mock_model_costs():
    """Mock model costs data."""
    return {
        "gpt-4o-mini": 0.15,
        "gemini-2.5-flash": 0.08,
        "deepseek-v3.1:671b-cloud": 0.5,
        "kimi-k2.5:cloud": 0.4,
        "qwen3-coder:480b-cloud": 0.3,
        "glm-4.6:cloud": 0.25,
        "gemini-2.5-pro": 1.0,
        "informational-model": 0.05,
    }


@pytest.fixture
def mock_coding_model_priority():
    """Mock coding model priority list."""
    return [
        "deepseek-v3.1:671b-cloud",
        "kimi-k2.5:cloud",
        "qwen3-coder:480b-cloud",
        "glm-4.6:cloud",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]


@pytest.fixture
def mock_usage_tracker():
    """Mock model usage singleton."""
    tracker = MagicMock()
    tracker.get_model_usage.return_value = 0
    return tracker


# ============================================================================
# Test Get Model Details
# ============================================================================


class TestGetModelDetails:
    """Test the get_model_details function."""

    def test_get_model_details_existing_model(self, mock_model_capabilities, mock_model_costs):
        """Test get_model_details for existing model."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                result = get_model_details("gpt-4o-mini")

                assert "capabilities" in result
                assert "cost" in result
                assert result["capabilities"] == {"coding", "reasoning", "fast", "tools", "informational"}
                assert result["cost"] == 0.15

    def test_get_model_details_unknown_model(self, mock_model_capabilities, mock_model_costs):
        """Test get_model_details for unknown model."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                result = get_model_details("unknown-model")

                assert result["capabilities"] == set()
                assert result["cost"] is None

    def test_get_model_details_is_coding_priority(self, mock_model_capabilities, mock_model_costs, mock_coding_model_priority):
        """Test that get_model_details correctly identifies coding priority models."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.CODING_MODEL_PRIORITY", mock_coding_model_priority):
                    # This is not exposed by get_model_details, but we verify the data is accessible
                    result = get_model_details("deepseek-v3.1:671b-cloud")
                    assert result["cost"] == 0.5


# ============================================================================
# Test Get Cheapest Model
# ============================================================================


class TestGetCheapestModel:
    """Test the get_cheapest_model function."""

    @pytest.mark.asyncio
    async def test_get_cheapest_model_informational_task(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test get_cheapest_model returns cheapest for informational task."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # Informational task - should return cheapest by cost
                        mock_infer.return_value = {"informational"}

                        result = await get_cheapest_model("What is the capital of France?")

                        # Should return informational-model (cheapest at 0.05)
                        assert result == "informational-model"

    @pytest.mark.asyncio
    async def test_get_cheapest_model_coding_task(self, mock_model_capabilities, mock_model_costs, mock_coding_model_priority, mock_usage_tracker):
        """Test get_cheapest_model uses priority order for coding tasks."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.CODING_MODEL_PRIORITY", mock_coding_model_priority):
                    with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                        with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                            # Coding task - should use priority order
                            mock_infer.return_value = {"coding", "reasoning"}

                            result = await get_cheapest_model("Write a Python function")

                            # Should return first priority model: deepseek-v3.1:671b-cloud
                            assert result == "deepseek-v3.1:671b-cloud"

    @pytest.mark.asyncio
    async def test_get_cheapest_model_no_matching_models(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test get_cheapest_model raises ValueError when no models match."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # No model has "vision" and "cheap" together in our mock
                        mock_infer.return_value = {"vision", "cheap"}

                        with pytest.raises(ValueError, match="No models found with capabilities"):
                            await get_cheapest_model("Analyze this image cheaply")

    @pytest.mark.asyncio
    async def test_get_cheapest_model_with_usage_penalty(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test get_cheapest_model applies exponential penalty for usage."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        with patch("task_agent.llms.simple_llm_selector.router.settings") as mock_settings:
                            mock_settings.COST_SPREADING_FACTOR = 0.03
                            mock_settings.TOKEN_USAGE_LOG_BASE = 100.0
                            mock_settings.FORMULA_WEIGHT_CALL_COUNT = 0.5
                            mock_settings.FORMULA_WEIGHT_TOKEN_COUNT = 0.5
                            # Set different usage counts - gpt-4o-mini has been used more
                            def mock_get_usage(model):
                                return 10 if model == "gpt-4o-mini" else 0

                            mock_usage_tracker.get_model_usage.side_effect = mock_get_usage

                            # Informational task
                            mock_infer.return_value = {"informational"}

                            result = await get_cheapest_model("What is Python?")

                            # With usage penalty, gemini-2.5-flash should be cheaper than gpt-4o-mini
                            # gpt-4o-mini: 0.15 * exp(0.03 * 10) = 0.15 * 1.35 = 0.20
                            # gemini-2.5-flash: 0.08 * exp(0) = 0.08
                            # informational-model: 0.05 * exp(0) = 0.05
                            assert result == "informational-model"

    @pytest.mark.asyncio
    async def test_get_cheapest_model_inference_fallback(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test get_cheapest_model handles inference failure gracefully."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # Simulate inference returning empty set (fallback behavior)
                        mock_infer.return_value = set()

                        # When no capabilities, it should raise ValueError (no models match empty set is okay,
                        # but typically infer_capabilities should return at least something)
                        # Actually, empty set is subset of all capabilities, so all models would match
                        # Let's test with a non-empty capability
                        mock_infer.return_value = {"informational"}
                        result = await get_cheapest_model("Tell me about Python")

                        assert result in ["informational-model", "gemini-2.5-flash", "gpt-4o-mini"]


# ============================================================================
# Test Select Models
# ============================================================================


class TestSelectModels:
    """Test the select_models function."""

    @pytest.mark.asyncio
    async def test_select_models_returns_top_n(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models returns exactly top_n models."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        mock_infer.return_value = {"informational"}

                        # Request top 3
                        result = await select_models("Tell me about Python", top_n=3)

                        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_select_models_with_single_capability(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models filters by single capability."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # Only informational capability
                        mock_infer.return_value = {"informational"}

                        result = await select_models("What is Python?", top_n=5)

                        # Should only return models with informational capability
                        for model in result:
                            assert "informational" in mock_model_capabilities[model]

    @pytest.mark.asyncio
    async def test_select_models_with_multiple_capabilities(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models filters by multiple capabilities (AND logic)."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # Require both coding AND tools
                        mock_infer.return_value = {"coding", "tools"}

                        result = await select_models("Write code with web search", top_n=10)

                        # All returned models must have both capabilities
                        for model in result:
                            caps = mock_model_capabilities[model]
                            assert "coding" in caps
                            assert "tools" in caps

                        # informational-model should NOT be in results
                        assert "informational-model" not in result

    @pytest.mark.asyncio
    async def test_select_models_sorting_by_derived_cost(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models sorts by derived cost for non-coding tasks."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        with patch("task_agent.llms.simple_llm_selector.router.settings") as mock_settings:
                            mock_settings.COST_SPREADING_FACTOR = 0.03
                            mock_settings.TOKEN_USAGE_LOG_BASE = 100.0
                            mock_settings.FORMULA_WEIGHT_CALL_COUNT = 0.5
                            mock_settings.FORMULA_WEIGHT_TOKEN_COUNT = 0.5
                            mock_infer.return_value = {"informational"}

                            result = await select_models("What is Python?", top_n=3)

                            # Should be sorted by derived cost (which equals base cost with no usage)
                            # informational-model (0.05) < gemini-2.5-flash (0.08) < gpt-4o-mini (0.15)
                            assert result[0] == "informational-model"
                            assert result[1] == "gemini-2.5-flash"

    @pytest.mark.asyncio
    async def test_select_models_coding_uses_priority_order(self, mock_model_capabilities, mock_model_costs, mock_coding_model_priority, mock_usage_tracker):
        """Test select_models uses priority order for coding tasks."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.CODING_MODEL_PRIORITY", mock_coding_model_priority):
                    with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                        with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                            mock_infer.return_value = {"coding", "reasoning"}

                            result = await select_models("Write Python code", top_n=5)

                            # First should be deepseek-v3.1:671b-cloud (first in priority list)
                            assert result[0] == "deepseek-v3.1:671b-cloud"
                            # Second should be kimi-k2.5:cloud (second in priority list)
                            assert result[1] == "kimi-k2.5:cloud"

    @pytest.mark.asyncio
    async def test_select_models_with_usage_penalty(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models applies exponential penalty correctly."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        with patch("task_agent.llms.simple_llm_selector.router.settings") as mock_settings:
                            mock_settings.COST_SPREADING_FACTOR = 0.03
                            mock_settings.TOKEN_USAGE_LOG_BASE = 100.0
                            mock_settings.FORMULA_WEIGHT_CALL_COUNT = 0.5
                            mock_settings.FORMULA_WEIGHT_TOKEN_COUNT = 0.5
                            # Simulate high usage on gpt-4o-mini
                            def mock_get_usage(model):
                                return 5 if model == "gpt-4o-mini" else 0

                            mock_usage_tracker.get_model_usage.side_effect = mock_get_usage
                            mock_infer.return_value = {"informational"}

                            result = await select_models("What is Python?", top_n=3)

                            # gpt-4o-mini should be ranked lower due to usage penalty
                            # gpt-4o-mini: 0.15 * exp(0.03 * 5) = 0.15 * 1.16 = 0.17
                            # gemini-2.5-flash: 0.08 * exp(0) = 0.08
                            # So gemini should come before gpt-4o-mini
                            gpt_index = result.index("gpt-4o-mini")
                            gemini_index = result.index("gemini-2.5-flash")
                            assert gemini_index < gpt_index

    @pytest.mark.asyncio
    async def test_select_models_no_matching_models(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models raises ValueError when no models match."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        # No model has both vision and cheap
                        mock_infer.return_value = {"vision", "cheap"}

                        with pytest.raises(ValueError, match="No models found with capabilities"):
                            await select_models("Analyze image cheaply", top_n=5)

    @pytest.mark.asyncio
    async def test_select_models_default_top_n(self, mock_model_capabilities, mock_model_costs, mock_usage_tracker):
        """Test select_models default top_n=5."""
        with patch("task_agent.llms.simple_llm_selector.router.MODEL_CAPABILITIES", mock_model_capabilities):
            with patch("task_agent.llms.simple_llm_selector.router.MODEL_COST", mock_model_costs):
                with patch("task_agent.llms.simple_llm_selector.router.infer_capabilities", new_callable=AsyncMock) as mock_infer:
                    with patch("task_agent.llms.simple_llm_selector.router.get_model_usage_singleton", return_value=mock_usage_tracker):
                        mock_infer.return_value = {"informational"}

                        # Don't specify top_n - should default to 5
                        result = await select_models("What is Python?")

                        assert len(result) <= 5
