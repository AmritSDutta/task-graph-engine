import logging
import time
from langchain_core.messages import AIMessage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),  # Be specific in production!
    reraise=True  # Raise the exception so the try/except block below catches it
)
async def call_llm_with_retry(llm, prompt) -> AIMessage:
    """Call LLM with retry logic, timing, and logging.

    Args:
        llm: The LangChain LLM instance to invoke
        prompt: The prompt to send to the LLM

    Returns:
        AIMessage: The response from the LLM
    """
    logger.info(f"Calling LLM: {type(llm).__name__}")
    start_time = time.time()

    try:
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
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"LLM call failed after {elapsed:.3f}s: {e}")
        raise
