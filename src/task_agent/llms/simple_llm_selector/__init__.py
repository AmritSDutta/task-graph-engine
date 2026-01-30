"""Simplified LLM Selector.

Uses a Groq model to infer task capabilities, then selects the best models
based on cost and capability matching.

Example:
    >>> from simple_llm_selector import select_models, get_cheapest_model
    >>>
    >>> # Get top 5 models
    >>> models = await select_models("Who invented calculus?")
    >>>
    >>> # Get just the cheapest
    >>> model = await get_cheapest_model("Quick API response needed")
"""

from .router import get_cheapest_model, get_model_details, select_models

__all__ = [
    "select_models",
    "get_cheapest_model",
    "get_model_details",
]
