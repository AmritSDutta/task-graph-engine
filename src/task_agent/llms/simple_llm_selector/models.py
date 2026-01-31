"""Model definitions for simplified LLM selector."""

import csv
import logging
from pathlib import Path
from typing import Literal, cast

from task_agent.config import settings
from task_agent.utils.path_utils import resolve_csv_path

logger = logging.getLogger(__name__)

# Capability tags
Capability = Literal[
    "reasoning",  # Complex reasoning, chain-of-thought
    "tools",  # Function calling, tool use
    "fast",  # Low latency
    "informational",  # General information, factual queries
    "coding",  # Code writing, programming, development
    "vision",  # Image understanding
    "long",  # Long context window
    "synthesizing",  # synthesizing
    "summarizing",  # summarizing
    "planning",  # planning
]


def _load_model_costs_from_csv(csv_path: str | Path | None = None) -> dict[str, float]:
    """
    Load model costs from CSV file.

    CSV format:
    model,cost
    gpt-4o-mini,0.035

    Args:
        csv_path: Path to CSV file. If None, uses MODEL_COST_CSV_PATH from config
            with smart resolution (absolute/relative/filename).

    Returns:
        Dictionary mapping model names to cost per 1M tokens.
    """
    if csv_path is None:
        # Use path from config with smart resolution
        csv_path = resolve_csv_path(settings.MODEL_COST_CSV_PATH)

    csv_path = Path(csv_path)
    costs: dict[str, float] = {}

    if not csv_path.exists():
        # Fallback to empty dict if CSV doesn't exist
        return costs

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            cost_str = row.get("cost", "0")

            # Handle empty or invalid cost values
            if cost_str:
                try:
                    costs[model_name] = float(cost_str)
                except ValueError:
                    # Skip invalid cost values
                    continue

    if costs:
        logger.info(
            f"✅ Loaded {len(costs)} model costs from {csv_path.name} "
            f"(range: ${min(costs.values()):.3f} - ${max(costs.values()):.1f} per 1M tokens)"
        )
        # Log top 5 cheapest models
        cheapest = sorted(costs.items(), key=lambda x: x[1])[:5]
        logger.info(f"💰 Cheapest models: {', '.join(f'{m} (${c:.3f})' for m, c in cheapest)}")
    else:
        logger.warning(f"⚠️  No model costs loaded from {csv_path}")

    return costs


def _load_model_capabilities_from_csv(csv_path: str | Path | None = None) -> dict[str, set[Capability]]:
    """
    Load model capabilities from CSV file.

    CSV format:
    model,reasoning,tools,fast,cheap,informational,coding,vision,long,synthesizing,summarizing,planning
    gpt-4o-mini,True,True,True,True,True,True,False,False,False,False

    Args:
        csv_path: Path to CSV file. If None, uses MODEL_CAPABILITY_CSV_PATH from config
            with smart resolution (absolute/relative/filename).

    Returns:
        Dictionary mapping model names to sets of capabilities.
    """
    if csv_path is None:
        # Use path from config with smart resolution
        csv_path = resolve_csv_path(settings.MODEL_CAPABILITY_CSV_PATH)

    csv_path = Path(csv_path)
    capabilities: dict[str, set[Capability]] = {}

    if not csv_path.exists():
        # Fallback to empty dict if CSV doesn't exist
        return capabilities

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model"]
            model_caps: set[Capability] = set()

            for cap, value in row.items():
                if cap == "model":
                    continue
                # Convert "True"/"False" strings to boolean
                # Handle None or empty values
                if value and value.lower() == "true":
                    model_caps.add(cast(Capability, cap))

            capabilities[model_name] = model_caps

    if capabilities:
        # Count capabilities per model
        cap_counts = {model: len(caps) for model, caps in capabilities.items()}
        logger.info(
            f"✅ Loaded capabilities for {len(capabilities)} models from {csv_path.name} "
            f"(avg: {sum(cap_counts.values()) / len(cap_counts):.1f} capabilities/model)"
        )
        # Log models with most capabilities
        most_capable = sorted(cap_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        logger.info(
            f"🎯 Most capable models: {', '.join(f'{m} ({c} caps)' for m, c in most_capable)}"
        )
    else:
        logger.warning(f"⚠️  No model capabilities loaded from {csv_path}")

    return capabilities


# Model capability matrix (loaded from CSV)
MODEL_CAPABILITIES: dict[str, set[Capability]] = _load_model_capabilities_from_csv()

# Preferred models for coding tasks (in priority order)
CODING_MODEL_PRIORITY = [
    "kimi-k2.5:cloud",
    "qwen3-coder:480b-cloud",
    "glm-4.6:cloud",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# Relative cost per 1M tokens (normalized, approximate)
# Lower is better. Loaded from CSV.
MODEL_COST: dict[str, float] = _load_model_costs_from_csv()


def get_model_capabilities(model: str) -> set[Capability]:
    """Get capabilities for a model."""
    return MODEL_CAPABILITIES.get(model, set())


def get_model_cost(model: str) -> float:
    """Get cost for a model."""
    return MODEL_COST.get(model, 999.0)
