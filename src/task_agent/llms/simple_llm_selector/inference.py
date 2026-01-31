"""Capability inference using LLM instead of rules."""

import logging

from .models import Capability
from ..llm_model_factory.llm_factory import create_llm
from ...config import settings


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
    planning_model: str = settings.INFERENCE_MODEL
    llm = create_llm(planning_model, temperature=0.0)

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
        - synthesizing: Synthesizing capabilities
        - summarizing: Summarizing capabilities
        - planning: Planning capabilities
        
        Task: "{task}"
        
        Rules:
        1. If task involves writing code, programming, or software: include "coding"
        2. If task asks for facts, explanations, or knowledge: include "informational"
        3. If task needs complex analysis: include "reasoning"
        4. Return only the required capability names as a comma-separated list.
        5. If unsure, default to "reasoning"
        
        Example outputs:
        coding, reasoning, cheap
        informational, cheap, long
        summarizing, synthesizing, long
        
        Task: "{task}"
        
        Response:
    """

    try:
        response = await llm.ainvoke(prompt)
        content = response.content.strip()
        logging.info(f"Capability inference[{planning_model}] suggested: {content}")

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

        return capabilities

    except Exception as e:
        # Fallback to informational on any error
        logging.info(f"Warning: Capability inference failed[{planning_model}]: {e}")
        return {"reasoning", "informational"}
