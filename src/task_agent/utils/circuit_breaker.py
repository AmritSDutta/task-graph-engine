import logging
import time

from langchain_core.messages import AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from task_agent.llms.llm_model_factory.llm_factory import create_llm
from task_agent.utils.model_live_usage import get_model_usage_singleton, ModelLiveUsage

logger = logging.getLogger(__name__)


def _extract_token_usage(response: AIMessage) -> dict | None:
    """Extract token usage information from LLM response.

    Args:
        response: The AIMessage response from the LLM

    Returns:
        dict with 'input_tokens', 'output_tokens', 'total_tokens' or None if not available
    """
    # Try usage_metadata first (LangChain 0.1+)
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        return {
            'input_tokens': response.usage_metadata.get('input_tokens', 0),
            'output_tokens': response.usage_metadata.get('output_tokens', 0),
            'total_tokens': response.usage_metadata.get('total_tokens', 0),
        }

    # Fall back to response_metadata (older LangChain style)
    if hasattr(response, 'response_metadata') and response.response_metadata:
        meta = response.response_metadata
        # Try common token usage keys
        if 'token_usage' in meta:
            tokens = meta['token_usage']
            return {
                'input_tokens': tokens.get('prompt_tokens', tokens.get('input_tokens', 0)),
                'output_tokens': tokens.get('completion_tokens', tokens.get('output_tokens', 0)),
                'total_tokens': tokens.get('total_tokens', 0),
            }
        # Direct keys in response_metadata
        return {
            'input_tokens': meta.get('prompt_tokens', meta.get('input_tokens', 0)),
            'output_tokens': meta.get('completion_tokens', meta.get('output_tokens', 0)),
            'total_tokens': meta.get('total_tokens', 0),
        }

    return None


async def _invoke_with_retry(llm, prompt) -> AIMessage:
    """Internal function with tenacity retry decorator.

    This function is decorated with @retry and will be called for both
    primary and fallback model attempts.
    """
    start_time = time.time()
    response = await llm.ainvoke(prompt)
    elapsed = time.time() - start_time

    # Extract token usage from response
    token_usage = _extract_token_usage(response)

    # Log completion with timing and token usage
    if token_usage:
        logger.info(
            f"LLM call completed in {elapsed:.3f}s | "
            f"Tokens: {token_usage.get('input_tokens', 'N/A')} in, "
            f"{token_usage.get('output_tokens', 'N/A')} out, "
            f"{token_usage.get('total_tokens', 'N/A')} total"
        )
    else:
        logger.info(f"LLM call completed in {elapsed:.3f}s")

    # Add timing metadata to response metadata if available
    if hasattr(response, 'metadata'):
        response.metadata['execution_time'] = f"{elapsed:.3f}s"

    return response


# Apply retry decorator to the internal function
_invoke_with_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),  # Be specific in production!
    reraise=True
)(_invoke_with_retry)


async def call_llm_with_retry(
        model_name: str,
        prompt,
        fallback_model: str | None = None,
        structured_output=None,
        **kwargs
) -> AIMessage:
    """Call LLM with retry logic, fallback model support, timing, and logging.

    Args:
        model_name: The model name to use (e.g., 'gpt-4o-mini')
        prompt: The prompt to send to the LLM
        fallback_model: Optional fallback model if primary fails (e.g., 'gpt-4o')
        structured_output: Optional Pydantic schema for structured output
        **kwargs: Additional arguments to pass to create_llm (e.g., temperature=0.0)

    Returns:
        AIMessage: The response from the LLM

    Raises:
        Exception: If both primary and fallback models fail after retries
    """
    # Try primary model
    logger.info(f"Calling LLM: {model_name}")
    llm = create_llm(model_name, **kwargs)
    if structured_output:
        llm = llm.with_structured_output(structured_output)

    model_usage: ModelLiveUsage = get_model_usage_singleton()

    try:
        model_usage.add_model_usage(model_name)
        return await _invoke_with_retry(llm, prompt)
    except Exception as e:
        logger.error(f"Primary model {model_name} failed after retries: {e}")

        # Try fallback model if provided
        if fallback_model:
            logger.warning(f"Falling back to {fallback_model}")
            llm_fallback = create_llm(fallback_model, **kwargs)
            if structured_output:
                llm_fallback = llm_fallback.with_structured_output(structured_output)
            try:
                return await _invoke_with_retry(llm_fallback, prompt)
            except Exception as fallback_error:
                logger.error(f"Fallback model {fallback_model} also failed: {fallback_error}")
                raise fallback_error
        raise
