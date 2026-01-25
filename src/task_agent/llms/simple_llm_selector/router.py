"""Simplified LLM router.

Flow:
1. Use Groq model to infer task capabilities
2. Filter models by capabilities
3. For coding: use priority order, otherwise sort by cost
4. Return top 5 candidates
"""

from typing import List

from .inference import infer_capabilities
from .models import MODEL_CAPABILITIES, MODEL_COST, CODING_MODEL_PRIORITY, Capability


async def select_models(task: str, top_n: int = 5) -> List[str]:
    """Select top N models for a given task.

    Selection process:
    1. Infer required capabilities using Groq LLM
    2. Filter models that have those capabilities
    3. If "coding" required: use priority order, then cost
    4. Otherwise: sort by cost (ascending)
    5. Return top N

    Args:
        task: The task description or question
        top_n: Number of top models to return (default: 5)

    Returns:
        List of model names, sorted appropriately

    Example:
        >>> models = await select_models("Write a Python function")
        >>> print(models)
        ['qwen3-coder:480b-cloud', 'glm-4.6:cloud', ...]
    """
    # Step 1: Infer capabilities from task
    print(f"Analyzing task: {task}")
    required_capabilities = await infer_capabilities(task)
    print(f"Required capabilities: {sorted(required_capabilities)}")
    print()

    # Step 2: Filter models by capabilities
    candidates = []
    for model, model_caps in MODEL_CAPABILITIES.items():
        # Check if model has all required capabilities
        if required_capabilities.issubset(model_caps):
            cost = MODEL_COST.get(model, 999.0)
            candidates.append((model, cost))

    if not candidates:
        raise ValueError(f"No models found with capabilities: {required_capabilities}")

    # Step 3: Sort based on capability type
    if "coding" in required_capabilities:
        # Use priority order for coding models
        print("Coding task detected - using priority model order")

        # Separate priority models from others
        priority_models = []
        other_models = []

        for model, cost in candidates:
            if model in CODING_MODEL_PRIORITY:
                # Sort by priority index (lower index = higher priority)
                priority_index = CODING_MODEL_PRIORITY.index(model)
                priority_models.append((priority_index, model, cost))
            else:
                other_models.append((model, cost))

        # Sort priority models by their defined order
        priority_models.sort(key=lambda x: x[0])
        # Sort other models by cost
        other_models.sort(key=lambda x: x[1])

        # Combine: priority models first (in order), then others by cost
        combined = [(model, cost) for _, model, cost in priority_models] + other_models
        candidates = combined

    else:
        # Non-coding: sort by cost (ascending)
        candidates.sort(key=lambda x: x[1])

    # Step 4: Return top N model names
    top_models = [model for model, _ in candidates[:top_n]]

    print(f"Found {len(candidates)} matching models, returning top {len(top_models)}:")
    for i, (model, cost) in enumerate(candidates[:top_n], 1):
        caps = MODEL_CAPABILITIES[model]
        # Check if this is a priority coding model
        priority_note = " [PRIORITY]" if model in CODING_MODEL_PRIORITY else ""
        print(f"  {i}. {model:35} cost: {cost:6.2f}{priority_note:12}  {sorted(caps)}")
    print()

    return top_models


async def get_cheapest_model(task: str) -> str:
    """Get the single cheapest model for a task.

    Args:
        task: The task description or question

    Returns:
        Name of the cheapest model that can handle the task
    """
    models = await select_models(task, top_n=1)
    return models[0]


def get_model_details(model: str) -> dict:
    """Get detailed information about a model.

    Args:
        model: Model name

    Returns:
        Dict with capabilities, cost, etc.
    """
    return {
        "capabilities": MODEL_CAPABILITIES.get(model, set()),
        "cost": MODEL_COST.get(model, None),
    }
