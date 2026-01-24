"""Demonstration of simplified LLM selector.

Uses Groq model for capability inference, then selects top models by cost.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from task_agent.llms.simple_llm_selector import select_models, get_cheapest_model
from task_agent.llms.llm_model_factory.llm_factory import create_llm


if __name__ == "__main__":
    print("=" * 70)
    print("Simplified LLM Selector Demo")
    print("=" * 70)
    print()

    # Demo 1: Select top models for calculus question
    task = "write a python function to find current date"

    models = select_models(task, top_n=5)

    print("-" * 70)
    print()

    # Demo 2: Get just the cheapest and execute
    print("Using cheapest model to answer:")
    print()

    cheapest = get_cheapest_model(task)
    print(f"Cheapest model: {cheapest}")
    print()

    # Create model and execute
    llm = create_llm(cheapest, temperature=0.0)
    response = llm.invoke(task)

    print("Answer:")
    print("-" * 70)
    print(response.content)
    print("-" * 70)