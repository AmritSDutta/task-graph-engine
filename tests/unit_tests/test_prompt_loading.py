"""Unit tests for prompt loading functionality.

Tests the external prompt file loading system in src/task_agent/llms/prompts/
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from task_agent.llms.prompts import (
    format_prompt,
    get_capability_inference_prompt,
    get_combiner_prompt,
    get_prompt,
    get_planner_prompt,
    get_subtask_prompt,
    list_available_prompts,
    load_prompt_template,
)


class TestLoadPromptTemplate:
    """Tests for load_prompt_template function."""

    def test_load_planner_prompt(self):
        """Test loading the planner prompt template."""
        template = load_prompt_template("planner")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "task planning assistant" in template.lower()
        assert "TODO list" in template

    def test_load_subtask_prompt(self):
        """Test loading the subtask prompt template."""
        template = load_prompt_template("subtask")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "helpful assistant" in template.lower()

    def test_load_combiner_prompt(self):
        """Test loading the combiner prompt template."""
        template = load_prompt_template("combiner")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "synthesizer" in template.lower()
        assert "{{user_query}}" in template

    def test_load_capability_inference_prompt(self):
        """Test loading the capability inference prompt template."""
        template = load_prompt_template("capability_inference")
        assert isinstance(template, str)
        assert len(template) > 0
        assert "task classifier" in template.lower()
        assert "{{task}}" in template

    def test_load_nonexistent_prompt_raises_error(self):
        """Test that loading a non-existent prompt raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt_template("nonexistent_prompt")
        assert "Prompt file not found" in str(exc_info.value)

    def test_load_nonexistent_prompt_shows_available_prompts(self):
        """Test that error message includes list of available prompts."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_prompt_template("nonexistent_prompt")
        # Should include some of the available prompts in error message
        error_msg = str(exc_info.value)
        assert "planner" in error_msg or "Available prompts:" in error_msg


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_format_single_variable(self):
        """Test formatting a prompt with a single variable."""
        template = "You are a {{role}} assistant."
        result = format_prompt(template, role="helpful")
        assert result == "You are a helpful assistant."

    def test_format_multiple_variables(self):
        """Test formatting a prompt with multiple variables."""
        template = "{{greeting}} {{name}}, you are a {{role}}."
        result = format_prompt(template, greeting="Hello", name="Alice", role="developer")
        assert result == "Hello Alice, you are a developer."

    def test_format_variable_reused(self):
        """Test formatting when the same variable appears multiple times."""
        template = "Task: {{task}}. Please complete {{task}} carefully."
        result = format_prompt(template, task="the assignment")
        assert result == "Task: the assignment. Please complete the assignment carefully."

    def test_format_with_missing_variable(self):
        """Test formatting when a variable in the template is not provided."""
        template = "You are a {{role}} assistant for {{domain}}."
        result = format_prompt(template, role="helpful")  # domain not provided
        # Should leave the missing variable as-is
        assert "You are a helpful assistant" in result
        assert "{{domain}}" in result or "domain" in result

    def test_format_empty_template(self):
        """Test formatting an empty template."""
        result = format_prompt("", role="helpful")
        assert result == ""

    def test_format_no_variables(self):
        """Test formatting a template with no variables."""
        template = "You are a helpful assistant."
        result = format_prompt(template, role="ignored")
        assert result == template

    def test_format_with_special_characters(self):
        """Test formatting with special characters in variable values."""
        template = "Query: {{query}}"
        result = format_prompt(template, query="SELECT * FROM users WHERE id = 1;")
        assert "SELECT * FROM users WHERE id = 1;" in result

    def test_format_with_newlines(self):
        """Test formatting preserves newlines."""
        template = "Line 1\n{{content}}\nLine 3"
        result = format_prompt(template, content="Line 2")
        assert result == "Line 1\nLine 2\nLine 3"

    def test_format_with_unicode(self):
        """Test formatting with unicode characters."""
        template = "Hello {{name}}, you're working on {{task}}"
        result = format_prompt(template, name="José", task="café")
        assert "José" in result
        assert "café" in result


class TestGetPrompt:
    """Tests for get_prompt convenience function."""

    def test_get_prompt_without_formatting(self):
        """Test get_prompt without any variables."""
        prompt = get_prompt("subtask")
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_prompt_with_formatting(self):
        """Test get_prompt with variable formatting."""
        prompt = get_prompt("combiner", user_query="Test query")
        assert "Test query" in prompt
        assert "{{user_query}}" not in prompt

    def test_get_prompt_nonexistent_raises_error(self):
        """Test get_prompt with non-existent prompt name."""
        with pytest.raises(FileNotFoundError):
            get_prompt("fake_prompt")


class TestListAvailablePrompts:
    """Tests for list_available_prompts function."""

    def test_list_prompts_returns_list(self):
        """Test that list_available_prompts returns a list."""
        prompts = list_available_prompts()
        assert isinstance(prompts, list)

    def test_list_prompts_contains_expected_prompts(self):
        """Test that all expected prompts are listed."""
        prompts = list_available_prompts()
        expected = {"planner", "subtask", "combiner", "capability_inference"}
        assert set(prompts) == expected

    def test_list_prompts_sorted(self):
        """Test that prompts are returned in alphabetical order."""
        prompts = list_available_prompts()
        assert prompts == sorted(prompts)


class TestConvenienceFunctions:
    """Tests for convenience getter functions."""

    def test_get_planner_prompt_returns_string(self):
        """Test get_planner_prompt returns non-empty string."""
        prompt = get_planner_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "task planning" in prompt.lower()

    def test_get_subtask_prompt_returns_string(self):
        """Test get_subtask_prompt returns non-empty string."""
        prompt = get_subtask_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "helpful assistant" in prompt.lower()

    def test_get_combiner_prompt_formats_query(self):
        """Test get_combiner_prompt formats user query."""
        query = "Why is the sky blue?"
        prompt = get_combiner_prompt(query)
        assert query in prompt
        assert "{{user_query}}" not in prompt

    def test_get_combiner_prompt_with_special_chars(self):
        """Test get_combiner_prompt with special characters in query."""
        query = "What's the effect of $100 price increase?"
        prompt = get_combiner_prompt(query)
        assert query in prompt

    def test_get_combiner_prompt_with_long_query(self):
        """Test get_combiner_prompt with a very long query."""
        query = "Explain " + "topic " * 100  # Long query
        prompt = get_combiner_prompt(query)
        assert query in prompt

    def test_get_capability_inference_prompt_formats_task(self):
        """Test get_capability_inference_prompt formats task."""
        task = "Write a Python function"
        prompt = get_capability_inference_prompt(task)
        assert task in prompt
        assert "{{task}}" not in prompt

    def test_get_capability_inference_prompt_contains_rules(self):
        """Test that capability inference prompt contains classification rules."""
        prompt = get_capability_inference_prompt("test task")
        assert "coding" in prompt.lower()
        assert "reasoning" in prompt.lower()
        assert "informational" in prompt.lower()


class TestPromptContent:
    """Tests for actual content of prompts."""

    def test_planner_prompt_has_json_structure(self):
        """Test that planner prompt mentions JSON output structure."""
        prompt = get_planner_prompt()
        assert "JSON" in prompt
        assert "title" in prompt
        assert "description" in prompt

    def test_planner_prompt_specifies_todo_count(self):
        """Test that planner prompt specifies TODO count range."""
        prompt = get_planner_prompt()
        assert "3-10" in prompt or "3" in prompt and "10" in prompt

    def test_capability_inference_prompt_has_capability_list(self):
        """Test that capability inference prompt lists all capabilities."""
        prompt = get_capability_inference_prompt("test")
        capabilities = [
            "reasoning",
            "tools",
            "fast",
            "cheap",
            "informational",
            "coding",
            "vision",
            "long",
            "synthesizing",
            "summarizing",
            "planning",
        ]
        for capability in capabilities:
            assert capability in prompt

    def test_subtask_prompt_mentions_tools(self):
        """Test that subtask prompt mentions tool availability."""
        prompt = get_subtask_prompt()
        assert "tool" in prompt.lower()


class TestPromptLoadingErrors:
    """Tests for error handling in prompt loading."""

    def test_load_prompt_with_invalid_path(self, tmp_path):
        """Test loading from an invalid prompts directory."""
        with patch("task_agent.llms.prompts._get_prompts_dir", return_value=tmp_path):
            with pytest.raises(FileNotFoundError):
                load_prompt_template("planner")

    def test_format_prompt_with_none_value(self):
        """Test formatting with None as variable value."""
        template = "Value: {{value}}"
        result = format_prompt(template, value=None)
        assert "Value: None" in result or "Value: " in result

    def test_format_prompt_with_empty_string(self):
        """Test formatting with empty string as variable value."""
        template = "Value: {{value}}"
        result = format_prompt(template, value="")
        assert result == "Value: "


class TestPromptEncoding:
    """Tests for file encoding handling."""

    def test_prompts_use_utf8_encoding(self):
        """Test that prompts are read with UTF-8 encoding."""
        # Load all prompts to ensure they're readable
        for prompt_name in list_available_prompts():
            template = load_prompt_template(prompt_name)
            assert isinstance(template, str)
            # Should not raise encoding errors
            assert len(template) > 0


class TestPromptDirectory:
    """Tests for prompt directory resolution."""

    def test_prompts_directory_exists(self):
        """Test that the prompts directory can be resolved."""
        from task_agent.llms.prompts import _get_prompts_dir

        prompts_dir = _get_prompts_dir()
        assert prompts_dir.exists()
        assert prompts_dir.is_dir()

    def test_prompts_directory_contains_prompt_files(self):
        """Test that prompts directory contains .prompt files."""
        from task_agent.llms.prompts import _get_prompts_dir

        prompts_dir = _get_prompts_dir()
        prompt_files = list(prompts_dir.glob("*.prompt"))
        assert len(prompt_files) >= 4  # At least the 4 main prompts

    def test_prompts_directory_is_cached(self):
        """Test that prompts directory location is cached."""
        from task_agent.llms.prompts import _get_prompts_dir

        dir1 = _get_prompts_dir()
        dir2 = _get_prompts_dir()
        assert dir1 == dir2
        # Should be the same object (cached)
        assert id(dir1) == id(dir2)
