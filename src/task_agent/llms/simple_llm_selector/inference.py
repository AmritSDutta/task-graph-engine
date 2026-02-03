"""Capability inference using LLM instead of rules."""

import logging
from time import sleep

from langchain_core.language_models import BaseChatModel

from .models import Capability
from ..llm_model_factory.llm_factory import create_llm
from ...config import settings
from ...llms.prompts import get_capability_inference_prompt


async def infer_capabilities(task: str) -> set[Capability]:
    """Infer required capabilities from task using an LLM.

    This uses a Groq model (fast & cheap) to analyze the task and determine
    which capabilities are required.

    Args:
        task: The task description or question

    Returns:
        Set of required capabilities
    """
    # Create Groq model for inference
    default_capabilities: set[Capability] = {"reasoning", "informational", "planning"}
    capability_inference_model: str = settings.INFERENCE_MODEL
    llm = create_llm(capability_inference_model, temperature=0.0)

    prompt = get_capability_inference_prompt(task)
    capabilities: set[Capability] = set()
    tried: int = 0
    while tried < settings.INFERENCE_MAX_RETRY:
        try:
            capabilities = await infer_capability_from_llm(capability_inference_model, llm, prompt)
            break
        except Exception as e:
            # Fallback to informational on any error
            logging.warning(f"Warning: Capability inference failed[{capability_inference_model}]: {e}")
            sleep(5.0)
            tried += 1

    if capabilities:
        return capabilities
    else:
        logging.info(f"Warning: returning default capabilities[{default_capabilities}]")
        return default_capabilities


async def infer_capability_from_llm(inference_model: str, llm: BaseChatModel, prompt: str) -> set[Capability]:
    response = await llm.ainvoke(prompt)
    content = response.content.strip()
    logging.info(f"Capability inference[{inference_model}] suggested: {content}")

    # Parse the response - look for capability names
    valid_caps = {
        "reasoning",  # Complex reasoning, chain-of-thought
        "tools",  # Function calling, tool use
        "fast",  # Low latency
        "cheap",  # Low cost per token
        "informational",  # General information, factual queries
        "coding",  # Code writing, programming, development
        "vision",  # Image understanding
        "long",  # Long context window
        "synthesizing",  # synthesizing
        "summarizing",  # summarizing
        "planning",  # planning
    }

    capabilities = set()

    # Split by comma and extract capabilities
    for cap in content.split(','):
        cap = cap.strip().strip('"\'').strip()
        if cap in valid_caps:
            capabilities.add(cap)

    # Fallback to informational if empty
    if not capabilities:
        capabilities.add("reasoning")

    logging.info(f"Inferred capabilities required for task: {capabilities}")

    return capabilities
