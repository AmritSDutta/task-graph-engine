"""Model definitions for simplified LLM selector."""

from typing import Literal

# Capability tags
Capability = Literal[
    "reasoning",     # Complex reasoning, chain-of-thought
    "tools",         # Function calling, tool use
    "fast",          # Low latency
    "cheap",         # Low cost per token
    "informational", # General information, factual queries
    "coding",        # Code writing, programming, development
    "vision",        # Image understanding
    "long",          # Long context window
    "synthesizing",  # synthesizing
    "summarizing",  # synthesizing
]

# Model capability matrix
MODEL_CAPABILITIES: dict[str, set[Capability]] = {
    # OpenAI models
    "gpt-4o": {"reasoning", "tools", "fast", "vision", "long", "informational", "coding"},
    "gpt-4o-mini": {"reasoning", "tools", "fast", "cheap", "informational", "coding"},
    "gpt-5-mini": {"reasoning", "tools", "fast", "cheap", "informational", "coding"},
    "gpt-5-nano": {"tools", "fast", "cheap", "informational", "summarizing", "synthesizing"},

    # Google models (2.5+)
    "gemini-2.5-flash": {"reasoning", "tools", "fast", "cheap", "vision", "long", "informational", "coding"},
    "gemini-2.5-flash-lite": {"reasoning", "tools", "fast", "cheap", "informational", "summarizing", "synthesizing"},
    "gemini-2.5-pro": {"reasoning", "tools", "vision", "long", "informational", "coding"},

    # Groq models
    "qwen/qwen-2.5-72b-instruct": {"reasoning", "tools", "fast", "cheap", "informational"},
    "qwen/qwen3-32b": {"reasoning", "tools", "fast", "cheap", "informational"},

    # ollama Cloud models
    "gemini-3-flash-preview:cloud": {"reasoning", "tools", "fast", "cheap", "vision", "long", "informational"},
    "qwen3-coder:480b-cloud": {"reasoning", "tools", "fast", "cheap", "informational", "coding"},
    "gemma3:27b-cloud": {"reasoning", "tools", "fast", "cheap", "informational", "summarizing", "synthesizing"},
    "glm-4.6:cloud": {"reasoning", "tools", "fast", "long", "informational", "coding"},
    "gpt-oss:20b-cloud": {"reasoning", "tools", "fast", "cheap", "informational", "summarizing", "synthesizing"},
}

# Preferred models for coding tasks (in priority order)
CODING_MODEL_PRIORITY = [
    "qwen3-coder:480b-cloud",
    "glm-4.6:cloud",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# Relative cost per 1M tokens (normalized, approximate)
# Lower is better.
MODEL_COST: dict[str, float] = {
    # OpenAI
    "gpt-4o": 5.0,
    "gpt-4o-mini": 0.6,
    "gpt-5-mini": 0.4,
    "gpt-5-nano": 0.2,

    # Google
    "gemini-2.5-flash": 0.08,
    "gemini-2.5-flash-lite": 0.05,
    "gemini-2.5-pro": 2.0,
    "gemini-3-flash-preview:cloud": 0.05,

    # Groq (very cheap)
    "qwen/qwen-2.5-72b-instruct": 0.05,
    "qwen/qwen3-32b": 0.05,
    "qwen3-coder:480b-cloud": 0.03,

    # Cloud
    "gemma3:27b-cloud": 0.03,
    "glm-4.6:cloud": 0.02,
    "gpt-oss:20b-cloud": 0.04,
}


def get_model_capabilities(model: str) -> set[Capability]:
    """Get capabilities for a model."""
    return MODEL_CAPABILITIES.get(model, set())


def get_model_cost(model: str) -> float:
    """Get cost for a model."""
    return MODEL_COST.get(model, 999.0)