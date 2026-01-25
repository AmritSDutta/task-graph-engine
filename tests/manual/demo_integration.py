"""Demonstration of Simple LLM Selector + LLM Factory integration.

This script shows how to use the simple selector to choose the best model
for a task, create it using the factory, and execute the query.
"""
import asyncio
import logging
import sys
from pathlib import Path

from task_agent.logging_config import setup_logging

# Add project root to path for imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task_agent.llms.simple_llm_selector import select_models, get_cheapest_model
from task_agent.llms.llm_model_factory.llm_factory import create_llm


async def main():
    setup_logging()
    logging.info("=" * 70)
    logging.info("Simple LLM Selector + Factory Integration Demo")
    logging.info("=" * 70)

    # Demo 1: Select top models for a task
    task = """
    You are a senior software architect and planner.

        Your task is to convert the following requirement into a clear, executable coding plan.
        
        Instructions:
        - Do NOT write code.
        - Produce a structured TODO list.
        - Break work into logical phases.
        - Each TODO item must be atomic and testable.
        - Explicitly list dependencies between steps.
        - Highlight risk areas or design decisions.
        - Assume production-quality standards.
        
        Context:
        <insert project context here>
        
        Task:
        <insert coding task or feature request here>
        
        Output Format:
        1. High-level objective
        2. Assumptions & constraints
        3. Architecture / design overview
        4. TODO list (numbered, ordered)
           - [ ] Task
             - Description
             - Inputs
             - Outputs
             - Dependencies
        5. Validation & testing plan
        6. Open questions / risks
        
        tasking building a small python function to check fibonacci numbers.
    """

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
