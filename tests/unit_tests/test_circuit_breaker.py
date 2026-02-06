"""Unit tests for circuit breaker - no API calls required."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage

from task_agent.utils.circuit_breaker import _extract_token_usage


class TestExtractTokenUsage:
    """Tests for _extract_token_usage function."""

    def test_extract_from_usage_metadata(self):
        """Test extraction from LangChain 0.1+ usage_metadata format."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['total_tokens'] == 150

    def test_extract_from_response_metadata_token_usage(self):
        """Test extraction from older response_metadata.token_usage format."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            }
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['total_tokens'] == 150

    def test_extract_from_response_metadata_direct_keys(self):
        """Test extraction from response_metadata with direct keys."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'input_tokens': 80,
            'output_tokens': 40,
            'total_tokens': 120
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 80
        assert result['output_tokens'] == 40
        assert result['total_tokens'] == 120

    def test_extract_with_prompt_tokens_fallback(self):
        """Test extraction with prompt_tokens fallback."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 75,
                'completion_tokens': 25,
                'total_tokens': 100
            }
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 75
        assert result['output_tokens'] == 25
        assert result['total_tokens'] == 100

    def test_extract_with_completion_tokens_fallback(self):
        """Test extraction with completion_tokens fallback."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'token_usage': {
                'input_tokens': 60,
                'completion_tokens': 30,
                'total_tokens': 90
            }
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 60
        assert result['output_tokens'] == 30
        assert result['total_tokens'] == 90

    def test_extract_with_zero_tokens(self):
        """Test extraction with zero token values."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 0
        assert result['output_tokens'] == 0
        assert result['total_tokens'] == 0

    def test_extract_with_missing_optional_fields(self):
        """Test extraction with missing optional token fields."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50
            # total_tokens is missing
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['total_tokens'] == 0  # defaults to 0

    def test_extract_returns_none_when_no_metadata(self):
        """Test that None is returned when no metadata is available."""
        response = AIMessage(content="Test response")
        # No usage_metadata or response_metadata

        result = _extract_token_usage(response)
        assert result is None

    def test_extract_returns_none_with_empty_usage_metadata(self):
        """Test that None is returned when usage_metadata is empty dict."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {}

        result = _extract_token_usage(response)
        assert result is None

    def test_extract_returns_none_with_empty_response_metadata(self):
        """Test that None is returned when response_metadata is empty dict."""
        response = AIMessage(content="Test response")
        response.response_metadata = {}

        result = _extract_token_usage(response)
        assert result is None

    def test_usage_metadata_takes_precedence(self):
        """Test that usage_metadata takes precedence over response_metadata."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }
        response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 200,
                'completion_tokens': 100,
                'total_tokens': 300
            }
        }

        result = _extract_token_usage(response)
        assert result is not None
        # Should use usage_metadata values
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 50
        assert result['total_tokens'] == 150

    def test_extract_with_large_token_counts(self):
        """Test extraction with large token counts."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 100000,
            'output_tokens': 50000,
            'total_tokens': 150000
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100000
        assert result['output_tokens'] == 50000
        assert result['total_tokens'] == 150000

    def test_extract_with_missing_hasattr_usage_metadata(self):
        """Test when response doesn't have usage_metadata attribute."""
        response = AIMessage(content="Test response")
        # Remove the attribute entirely
        if hasattr(response, 'usage_metadata'):
            delattr(response, 'usage_metadata')

        result = _extract_token_usage(response)
        assert result is None

    def test_extract_with_missing_hasattr_response_metadata(self):
        """Test when response doesn't have response_metadata attribute."""
        response = AIMessage(content="Test response")
        response.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }
        # Remove response_metadata attribute
        if hasattr(response, 'response_metadata'):
            delattr(response, 'response_metadata')

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100


class TestExtractTokenUsageEdgeCases:
    """Tests for edge cases in token usage extraction."""

    def test_extract_with_none_usage_metadata(self):
        """Test when usage_metadata exists but is None."""
        response = AIMessage(content="Test response")
        response.usage_metadata = None

        result = _extract_token_usage(response)
        assert result is None

    def test_extract_with_none_response_metadata(self):
        """Test when response_metadata exists but is None."""
        response = AIMessage(content="Test response")
        response.response_metadata = None

        result = _extract_token_usage(response)
        assert result is None

    def test_extract_with_partial_token_usage(self):
        """Test when token_usage dict has missing keys."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'token_usage': {
                'prompt_tokens': 100
                # Missing completion_tokens and total_tokens
            }
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 100
        assert result['output_tokens'] == 0  # defaults to 0

    def test_extract_with_mixed_token_formats(self):
        """Test handling of different token format names."""
        response = AIMessage(content="Test response")
        response.response_metadata = {
            'input_tokens': 60,
            'completion_tokens': 30,  # mixed format
            'total_tokens': 90
        }

        result = _extract_token_usage(response)
        assert result is not None
        assert result['input_tokens'] == 60
        assert result['output_tokens'] == 30
        assert result['total_tokens'] == 90


class TestInvokeWithRetryBehavior:
    """Tests for _invoke_with_retry behavior (non-LLM tests)."""

    @pytest.mark.asyncio
    async def test_invoke_with_retry_adds_timing_metadata(self):
        """Test that execution time is added to response metadata."""
        from task_agent.utils.circuit_breaker import _invoke_with_retry

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="Test")
        mock_response.metadata = {}
        mock_llm.ainvoke.return_value = mock_response

        with patch('task_agent.utils.circuit_breaker.time.time', side_effect=[0.0, 1.5]):
            result = await _invoke_with_retry(mock_llm, "test prompt", model_name="gpt-4o-mini")

        # Verify metadata was added
        assert 'execution_time' in result.metadata
        assert result.metadata['execution_time'] == "1.500s"

    @pytest.mark.asyncio
    async def test_invoke_with_retry_extracts_token_usage(self):
        """Test that token usage is extracted and logged."""
        from task_agent.utils.circuit_breaker import _invoke_with_retry

        # Mock LLM with token usage
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="Test")
        mock_response.usage_metadata = {
            'input_tokens': 100,
            'output_tokens': 50,
            'total_tokens': 150
        }
        mock_llm.ainvoke.return_value = mock_response

        with patch('task_agent.utils.circuit_breaker.time.time', return_value=0.0):
            result = await _invoke_with_retry(mock_llm, "test prompt", model_name="gpt-4o-mini")

        # Verify response is returned
        assert result.content == "Test"

    @pytest.mark.asyncio
    async def test_invoke_with_retry_handles_missing_metadata_attribute(self):
        """Test handling when response doesn't have metadata attribute."""
        from task_agent.utils.circuit_breaker import _invoke_with_retry

        # Mock LLM
        mock_llm = AsyncMock()
        mock_response = AIMessage(content="Test")
        # Remove metadata attribute if it exists
        if hasattr(mock_response, 'metadata'):
            delattr(mock_response, 'metadata')
        mock_llm.ainvoke.return_value = mock_response

        with patch('task_agent.utils.circuit_breaker.time.time', return_value=0.0):
            result = await _invoke_with_retry(mock_llm, "test prompt", model_name="gpt-4o-mini")

        # Should not raise error
        assert result.content == "Test"


class TestCallLlmWithRetryBehavior:
    """Tests for call_llm_with_retry behavior (non-LLM tests)."""

    @pytest.mark.asyncio
    async def test_call_llm_with_retry_creates_llm_with_kwargs(self):
        """Test that additional kwargs are passed to create_llm."""
        from task_agent.utils.circuit_breaker import call_llm_with_retry

        with patch('task_agent.utils.circuit_breaker.create_llm') as mock_create:
            # Create a proper async mock
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_llm
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test"))
            mock_create.return_value = mock_llm

            # Call with additional kwargs
            await call_llm_with_retry(
                "gpt-4o",
                "test prompt",
                temperature=0.7,
                max_tokens=1000
            )

            # Verify create_llm was called with kwargs
            mock_create.assert_called_once()
            args, kwargs = mock_create.call_args
            assert args[0] == "gpt-4o"
            assert kwargs.get('temperature') == 0.7
            assert kwargs.get('max_tokens') == 1000

    @pytest.mark.asyncio
    async def test_call_llm_with_retry_with_structured_output(self):
        """Test that structured_output is applied to LLM."""
        from task_agent.utils.circuit_breaker import call_llm_with_retry

        with patch('task_agent.utils.circuit_breaker.create_llm') as mock_create:
            # Create a proper async mock
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_llm
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test"))
            mock_create.return_value = mock_llm

            # Call with structured output
            from pydantic import BaseModel

            class TestSchema(BaseModel):
                name: str

            await call_llm_with_retry(
                "gpt-4o",
                "test prompt",
                structured_output=TestSchema
            )

            # Verify with_structured_output was called
            mock_llm.with_structured_output.assert_called_once_with(TestSchema)

    @pytest.mark.asyncio
    async def test_call_llm_with_retry_without_structured_output(self):
        """Test that LLM is used directly when no structured output."""
        from task_agent.utils.circuit_breaker import call_llm_with_retry

        with patch('task_agent.utils.circuit_breaker.create_llm') as mock_create:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test"))
            mock_create.return_value = mock_llm

            # Call without structured output
            await call_llm_with_retry("gpt-4o", "test prompt")

            # Verify with_structured_output was NOT called
            mock_llm.with_structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_call_llm_with_retry_logs_model_name(self):
        """Test that model name is logged."""
        from task_agent.utils.circuit_breaker import call_llm_with_retry
        import logging

        with patch('task_agent.utils.circuit_breaker.create_llm') as mock_create:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="Test"))
            mock_create.return_value = mock_llm

            with patch('task_agent.utils.circuit_breaker.logger') as mock_logger:
                await call_llm_with_retry("gemini-2.5-flash", "test prompt")

                # Verify logging occurred
                mock_logger.info.assert_any_call("Calling LLM: gemini-2.5-flash")
