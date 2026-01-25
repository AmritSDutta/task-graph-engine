"""Demonstration of simplified LLM selector.

Uses Groq model for capability inference, then selects top models by cost.
"""

import asyncio
import logging
import sys
from pathlib import Path

from task_agent.logging_config import setup_logging

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from task_agent.llms.simple_llm_selector import select_models, get_cheapest_model
from task_agent.llms.llm_model_factory.llm_factory import create_llm


async def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("Simplified LLM Selector Demo (Async)")
    logging.info("=" * 70)

    # Demo 1: Select top models for calculus question
    task = "write a python function to find current date"

    models = await select_models(task, top_n=5)

    logging.info("-" * 70)

    # Demo 2: Get just the cheapest and execute
    logging.info("Using cheapest model to answer:")

    cheapest = await get_cheapest_model(task)
    logging.info(f"Cheapest model: {cheapest}")

    # Create model and execute
    llm = create_llm(cheapest, temperature=0.0)
    response = await llm.ainvoke(task)

    logging.info("Answer:")
    logging.info("-" * 70)
    logging.info(response.content)
    logging.info("-" * 70)


if __name__ == "__main__":
    asyncio.run(main())
