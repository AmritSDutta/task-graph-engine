"""Prompt management for external .prompt files.

This module provides functions to load system prompts from external files,
enabling easy editing, versioning, and templating.

Benefits:
- Non-technical team can edit prompts
- Git version control
- Easy A/B testing
- Template parameterization with {{variable}} syntax
"""

import logging
import os
from pathlib import Path
from typing import Any

# Module-level directory cache
_PROMPTS_DIR: Path | None = None


def _get_prompts_dir() -> Path:
    """Get the prompts directory, caching the result.

    The prompts directory is relative to this module's location.
    """
    global _PROMPTS_DIR

    if _PROMPTS_DIR is None:
        # Get the directory where this __init__.py file is located
        current_dir = Path(__file__).parent.resolve()
        _PROMPTS_DIR = current_dir

    return _PROMPTS_DIR


def load_prompt_template(name: str) -> str:
    """Load a prompt template from a .prompt file.

    Args:
        name: Name of the prompt file without extension (e.g., "planner", "combiner")

    Returns:
        The prompt template as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
        IOError: If there's an error reading the file

    Example:
        >>> template = load_prompt_template("planner")
        >>> print(template)
        'You are a task planning assistant...\\n\\nGenerate {{count}} TODO items...'
    """
    prompts_dir = _get_prompts_dir()
    prompt_path = prompts_dir / f"{name}.prompt"

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}\n"
            f"Available prompts: {list_available_prompts()}"
        )

    try:
        with open(prompt_path, encoding="utf-8") as f:
            template = f.read()
        logging.debug(f"Loaded prompt template: {name} from {prompt_path}")
        return template
    except Exception as e:
        logging.error(f"Error reading prompt file {prompt_path}: {e}")
        raise IOError(f"Failed to read prompt file: {e}") from e


def format_prompt(template: str, **kwargs: Any) -> str:
    """Format a prompt template with variables.

    Uses {{variable}} syntax for template parameters.

    Args:
        template: The prompt template string with {{variable}} placeholders
        **kwargs: Variable values to substitute

    Returns:
        The formatted prompt string

    Example:
        >>> template = "You are a {{role}} assistant for {{domain}}."
        >>> formatted = format_prompt(template, role="helpful", domain="coding")
        >>> print(formatted)
        'You are a helpful assistant for coding.'
    """
    try:
        # Use str.format with custom braces for {{var}} syntax
        # Replace {{var}} with {var} for Python's str.format
        formatted = template
        for key, value in kwargs.items():
            placeholder = f"{{{{{key}}}}}"  # {{key}}
            formatted = formatted.replace(placeholder, str(value))

        return formatted
    except Exception as e:
        logging.error(f"Error formatting prompt: {e}")
        # Return template with placeholders intact on error
        return template


def get_prompt(name: str, **kwargs: Any) -> str:
    """Load and optionally format a prompt template.

    This is a convenience function that combines load_prompt_template and format_prompt.

    Args:
        name: Name of the prompt file without extension
        **kwargs: Optional variables for template formatting

    Returns:
        The loaded and formatted prompt string

    Example:
        >>> # Without formatting
        >>> prompt = get_prompt("subtask")
        >>>
        >>> # With formatting
        >>> prompt = get_prompt("combiner", user_query="Write a Python function")
    """
    template = load_prompt_template(name)

    if kwargs:
        return format_prompt(template, **kwargs)

    return template


def list_available_prompts() -> list[str]:
    """List all available prompt files.

    Returns:
        List of prompt names (without .prompt extension)

    Example:
        >>> list_available_prompts()
        ['planner', 'subtask', 'combiner', 'capability_inference']
    """
    prompts_dir = _get_prompts_dir()

    if not prompts_dir.exists():
        logging.warning(f"Prompts directory does not exist: {prompts_dir}")
        return []

    prompt_files = list(prompts_dir.glob("*.prompt"))
    prompt_names = [f.stem for f in prompt_files if f.is_file()]

    return sorted(prompt_names)


# Convenience exports for common prompts
def get_planner_prompt() -> str:
    """Get the planner prompt template."""
    return get_prompt("planner")


def get_subtask_prompt() -> str:
    """Get the subtask prompt template."""
    return get_prompt("subtask")


def get_combiner_prompt(user_query: str) -> str:
    """Get the combiner prompt template with user query formatted.

    Args:
        user_query: The original user request

    Returns:
        Formatted combiner prompt
    """
    return get_prompt("combiner", user_query=user_query)


def get_capability_inference_prompt(task: str) -> str:
    """Get the capability inference prompt template with task formatted.

    Args:
        task: The task to analyze

    Returns:
        Formatted capability inference prompt
    """
    return get_prompt("capability_inference", task=task)


__all__ = [
    "load_prompt_template",
    "format_prompt",
    "get_prompt",
    "list_available_prompts",
    "get_planner_prompt",
    "get_subtask_prompt",
    "get_combiner_prompt",
    "get_capability_inference_prompt",
]
