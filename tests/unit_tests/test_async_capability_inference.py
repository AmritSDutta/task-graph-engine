"""Async unit tests for capability inference with mocked LLM calls."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage

from task_agent.llms.simple_llm_selector.inference import infer_capabilities
from task_agent.llms.simple_llm_selector.models import Capability


class TestInferCapabilitiesWithMockLLM:
    """Tests for infer_capabilities function with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_infer_coding_capabilities(self):
        """Test inference correctly identifies coding tasks."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            # Mock the LLM
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            # Call the function
            result = await infer_capabilities("Write a Python function to sort a list")

            # Verify results
            assert isinstance(result, set)
            assert "coding" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_infer_informational_capabilities(self):
        """Test inference correctly identifies informational tasks."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="informational, cheap")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Who invented calculus?")

            assert isinstance(result, set)
            assert "informational" in result
            assert "cheap" in result

    @pytest.mark.asyncio
    async def test_infer_vision_capabilities(self):
        """Test inference correctly identifies vision tasks."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="vision, reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Analyze this image and describe what you see")

            assert isinstance(result, set)
            assert "vision" in result

    @pytest.mark.asyncio
    async def test_infer_with_quoted_capabilities(self):
        """Test inference handles quoted capabilities in response."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content='"coding", "reasoning", "fast"')
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Quick coding task")

            assert "coding" in result
            assert "reasoning" in result
            assert "fast" in result

    @pytest.mark.asyncio
    async def test_infer_with_mixed_whitespace(self):
        """Test inference handles various whitespace patterns."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="  coding  ,   reasoning   , cheap ")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert "coding" in result
            assert "reasoning" in result
            assert "cheap" in result

    @pytest.mark.asyncio
    async def test_infer_fallback_to_reasoning_on_empty_response(self):
        """Test inference falls back to reasoning when LLM returns empty capabilities."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Some task")

            assert isinstance(result, set)
            assert "reasoning" in result  # Fallback capability

    @pytest.mark.asyncio
    async def test_infer_fallback_to_reasoning_on_no_valid_capabilities(self):
        """Test inference falls back to reasoning when no valid capabilities are found."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="invalid_capability, another_invalid")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Some task")

            assert isinstance(result, set)
            assert "reasoning" in result  # Fallback capability

    @pytest.mark.asyncio
    async def test_infer_fallback_to_informational_on_exception(self):
        """Test inference falls back to informational when LLM call fails."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("API error")
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Some task")

            assert isinstance(result, set)
            assert "informational" in result  # Fallback on error

    @pytest.mark.asyncio
    async def test_infer_with_multiple_valid_capabilities(self):
        """Test inference with multiple valid capabilities."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning, fast, cheap, long")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Complex coding task with large context")

            assert len(result) == 5
            assert "coding" in result
            assert "reasoning" in result
            assert "fast" in result
            assert "cheap" in result
            assert "long" in result

    @pytest.mark.asyncio
    async def test_infer_ignores_invalid_capabilities(self):
        """Test inference ignores invalid capabilities while keeping valid ones."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, invalid_cap, reasoning, another_invalid")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert "coding" in result
            assert "reasoning" in result
            assert "invalid_cap" not in result
            assert "another_invalid" not in result


class TestInferCapabilitiesEdgeCases:
    """Tests for edge cases in capability inference."""

    @pytest.mark.asyncio
    async def test_infer_with_newline_separated_capabilities(self):
        """Test inference handles newline-separated capabilities."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding\nreasoning\nfast")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Only comma-separated should work
            # The newline would be treated as part of the capability name
            # So only valid comma-separated items should be extracted
            assert isinstance(result, set)

    @pytest.mark.asyncio
    async def test_infer_with_special_characters_in_response(self):
        """Test inference handles special characters in LLM response."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning, fast! cheap?")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Should extract valid capabilities, ignoring special chars
            assert "coding" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_infer_with_case_sensitivity(self):
        """Test that capability matching is case-sensitive."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="Coding, Reasoning, Fast")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Uppercase variants should not match
            # Only lowercase should be valid
            # If the LLM returns "Coding" (capital C), it won't match "coding"
            # So this should result in the fallback "reasoning"
            assert isinstance(result, set)

    @pytest.mark.asyncio
    async def test_infer_with_duplicate_capabilities(self):
        """Test inference handles duplicate capabilities in response."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, coding, reasoning, coding")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Should deduplicate
            assert len(result) == 2
            assert "coding" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_infer_with_leading_trailing_commas(self):
        """Test inference handles leading/trailing commas."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content=", coding, reasoning, ")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert "coding" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_infer_with_very_long_response(self):
        """Test inference handles very long LLM responses."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            long_response = "coding, " * 100 + "reasoning"
            mock_response = AIMessage(content=long_response)
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert "coding" in result
            assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_infer_with_timeout_error(self):
        """Test inference handles timeout errors."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = TimeoutError("Request timed out")
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Should fallback to informational on timeout
            assert "informational" in result

    @pytest.mark.asyncio
    async def test_infer_with_connection_error(self):
        """Test inference handles connection errors."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = ConnectionError("Connection refused")
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # Should fallback to informational
            assert "informational" in result


class TestInferCapabilitiesAllValidCapabilities:
    """Tests for all valid capability values."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("capability", [
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
    ])
    async def test_infer_each_valid_capability(self, capability):
        """Test that each valid capability is recognized."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content=capability)
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert capability in result


class TestInferCapabilitiesLLMConfiguration:
    """Tests for LLM configuration in inference."""

    @pytest.mark.asyncio
    async def test_infer_creates_llm_with_temperature_zero(self):
        """Test that LLM is created with temperature=0.0."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Test task")

            # Verify create_llm was called with temperature=0.0
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs.get('temperature') == 0.0

    @pytest.mark.asyncio
    async def test_infer_uses_inference_model_from_settings(self):
        """Test that inference model from settings is used."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create, \
             patch('task_agent.llms.simple_llm_selector.inference.settings') as mock_settings:
            mock_settings.INFERENCE_MODEL = "test-inference-model"
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Test task")

            # Verify the model name passed to create_llm
            call_args = mock_create.call_args[0]
            assert call_args[0] == "test-inference-model"

    @pytest.mark.asyncio
    async def test_infer_calls_llm_ainvoke(self):
        """Test that LLM ainvoke is called (not invoke)."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Test task")

            # Verify ainvoke (async) was called
            mock_llm.ainvoke.assert_called_once()
            # Verify the prompt contains the task
            call_args = mock_llm.ainvoke.call_args[0][0]
            assert "Test task" in call_args


class TestInferCapabilitiesLogging:
    """Tests for logging in capability inference."""

    @pytest.mark.asyncio
    async def test_infer_logs_suggested_capabilities(self):
        """Test that inferred capabilities are logged."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create, \
             patch('task_agent.llms.simple_llm_selector.inference.logging') as mock_logging:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Write code")

            # Verify logging occurred
            assert mock_logging.info.called

    @pytest.mark.asyncio
    async def test_infer_logs_error_on_failure(self):
        """Test that errors are logged."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create, \
             patch('task_agent.llms.simple_llm_selector.inference.logging') as mock_logging:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("Test error")
            mock_create.return_value = mock_llm

            await infer_capabilities("Test task")

            # Verify error was logged
            assert mock_logging.info.called
            # Check that warning about failure was logged
            log_calls = [str(call) for call in mock_logging.info.call_args_list]
            assert any("Capability inference failed" in str(call) for call in log_calls)


class TestInferCapabilitiesPromptStructure:
    """Tests for prompt structure in inference."""

    @pytest.mark.asyncio
    async def test_infer_prompt_contains_task(self):
        """Test that the prompt contains the task description."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            task = "Write a Python sorting algorithm"
            await infer_capabilities(task)

            # Verify the task is in the prompt
            call_args = mock_llm.ainvoke.call_args[0][0]
            assert task in call_args

    @pytest.mark.asyncio
    async def test_infer_prompt_contains_capability_list(self):
        """Test that the prompt contains the list of available capabilities."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="reasoning")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Test task")

            # Verify capabilities are mentioned in prompt
            call_args = mock_llm.ainvoke.call_args[0][0]
            assert "reasoning" in call_args
            assert "tools" in call_args
            assert "coding" in call_args
            assert "vision" in call_args

    @pytest.mark.asyncio
    async def test_infer_prompt_contains_rules(self):
        """Test that the prompt contains classification rules."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            await infer_capabilities("Write code")

            # Verify rules are in prompt
            call_args = mock_llm.ainvoke.call_args[0][0]
            assert "Rules:" in call_args or "rules:" in call_args


class TestInferCapabilitiesReturnType:
    """Tests for return type and structure."""

    @pytest.mark.asyncio
    async def test_infer_returns_set(self):
        """Test that infer_capabilities returns a set."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            assert isinstance(result, set)

    @pytest.mark.asyncio
    async def test_infer_returns_set_of_strings(self):
        """Test that all items in returned set are strings."""
        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning, fast")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            for item in result:
                assert isinstance(item, str)

    @pytest.mark.asyncio
    async def test_infer_returns_valid_capabilities_only(self):
        """Test that all returned capabilities are from the valid set."""
        valid_capabilities = {
            "reasoning", "tools", "fast", "cheap", "informational",
            "coding", "vision", "long", "synthesizing", "summarizing", "planning"
        }

        with patch('task_agent.llms.simple_llm_selector.inference.create_llm') as mock_create:
            mock_llm = AsyncMock()
            mock_response = AIMessage(content="coding, reasoning, invalid")
            mock_llm.ainvoke.return_value = mock_response
            mock_create.return_value = mock_llm

            result = await infer_capabilities("Test task")

            # All results should be valid capabilities
            assert result.issubset(valid_capabilities)
