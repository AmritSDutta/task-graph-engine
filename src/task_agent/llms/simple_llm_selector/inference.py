"""Capability inference using LLM instead of rules."""

import logging

from .models import Capability
from ..llm_model_factory.llm_factory import create_llm

# Use a Groq model for fast, cheap inference
INFERENCE_MODEL = "qwen/qwen3-32b"


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
    llm = create_llm(INFERENCE_MODEL, temperature=0.0)

    prompt = f"""You are a task classifier. Analyze this task and identify which LLM capabilities are required.

        Available capabilities:
        - reasoning: Complex reasoning, chain-of-thought, analysis
        - tools: Function calling, tool use, API interactions
        - fast: Low latency, quick response time
        - cheap: Low cost per token (budget-conscious)
        - informational: General information, factual queries, knowledge retrieval
        - coding: Code writing, programming, software development
        - vision: Image understanding, visual content
        - long: Long context window needed
        
        Task: "{task}"
        
        Rules:
        1. If task involves writing code, programming, or software: include "coding"
        2. If task asks for facts, explanations, or knowledge: include "informational"
        3. If task needs complex analysis: include "reasoning"
        4. Return only the required capability names as a comma-separated list.
        5. If unsure, default to "informational"
        
        Example outputs:
        coding, reasoning
        informational
        coding, tools
        
        Task: "{task}"
        
        Response:
    """

    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip()

        # Parse the response - look for capability names
        valid_caps = {
            "reasoning", "tools", "fast", "cheap",
            "informational", "coding", "vision", "long"
        }

        capabilities = set()

        # Split by comma and extract capabilities
        for cap in content.split(','):
            cap = cap.strip().strip('"\'').strip()
            if cap in valid_caps:
                capabilities.add(cap)

        # Fallback to informational if empty
        if not capabilities:
            capabilities.add("informational")

        return capabilities

    except Exception as e:
        # Fallback to informational on any error
        logging.info(f"Warning: Capability inference failed: {e}")
        return {"informational"}
