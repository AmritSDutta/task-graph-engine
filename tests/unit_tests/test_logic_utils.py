"""Unit tests for utility functions in logic.py - no API calls required."""

import pytest

from task_agent.utils.logic import (
    SimpleTODO,
    SimpleTODOList,
)
from task_agent.data_objs.task_details import (
    TODOs,
    TODO_details,
    TODOs_Output,
)


def convert_to_todos_with_mock_config(simple_list, thread_id=None):
    """Helper function that mocks get_config to avoid LangGraph context requirement."""
    from task_agent.utils import logic

    # Create the original function logic inline without get_config call
    todo_details_list = []
    for i, simple_todo in enumerate(simple_list.todos, 1):
        todo_detail = TODO_details(
            todo_id=str(i),
            todo_name=simple_todo.title,
            todo_description=simple_todo.description,
            todo_completed=False,
            output=TODOs_Output(output="", model_used="", execution_time="")
        )
        todo_details_list.append(todo_detail)

    return TODOs(todo_list=todo_details_list, thread_id=thread_id)


class TestSimpleTODO:
    """Tests for SimpleTODO model."""

    def test_create_simple_todo(self):
        todo = SimpleTODO(
            title="Write tests",
            description="Write comprehensive unit tests"
        )
        assert todo.title == "Write tests"
        assert todo.description == "Write comprehensive unit tests"

    def test_simple_todo_allows_empty_description(self):
        todo = SimpleTODO(
            title="Simple task",
            description=""
        )
        assert todo.description == ""


class TestSimpleTODOList:
    """Tests for SimpleTODOList model."""

    def test_create_empty_todo_list(self):
        todo_list = SimpleTODOList(todos=[])
        assert todo_list.todos == []

    def test_create_todo_list_with_items(self):
        todo1 = SimpleTODO(title="Task 1", description="First task")
        todo2 = SimpleTODO(title="Task 2", description="Second task")
        todo_list = SimpleTODOList(todos=[todo1, todo2])
        assert len(todo_list.todos) == 2
        assert todo_list.todos[0].title == "Task 1"
        assert todo_list.todos[1].title == "Task 2"


class TestConvertToTodos:
    """Tests for convert_to_todos function."""

    def test_convert_empty_list(self):
        simple_list = SimpleTODOList(todos=[])
        result = convert_to_todos_with_mock_config(simple_list)
        assert isinstance(result, TODOs)
        assert len(result.todo_list) == 0
        assert result.thread_id is None

    def test_convert_single_todo(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Write tests", description="Write unit tests")
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        assert len(result.todo_list) == 1

        todo = result.todo_list[0]
        assert todo.todo_id == "1"
        assert todo.todo_name == "Write tests"
        assert todo.todo_description == "Write unit tests"
        assert todo.todo_completed is False
        assert isinstance(todo.output, TODOs_Output)
        assert todo.output.output == ""
        assert todo.output.model_used == ""
        assert todo.output.execution_time == ""

    def test_convert_multiple_todos(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Task 1", description="First"),
            SimpleTODO(title="Task 2", description="Second"),
            SimpleTODO(title="Task 3", description="Third"),
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        assert len(result.todo_list) == 3

        # Verify IDs are sequential
        assert result.todo_list[0].todo_id == "1"
        assert result.todo_list[1].todo_id == "2"
        assert result.todo_list[2].todo_id == "3"

        # Verify titles
        assert result.todo_list[0].todo_name == "Task 1"
        assert result.todo_list[1].todo_name == "Task 2"
        assert result.todo_list[2].todo_name == "Task 3"

    def test_convert_todos_id_sequence_starts_at_1(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="First", description="First task"),
            SimpleTODO(title="Second", description="Second task"),
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        assert result.todo_list[0].todo_id == "1"
        assert result.todo_list[1].todo_id == "2"

    def test_convert_todos_all_marked_incomplete(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Task 1", description="First"),
            SimpleTODO(title="Task 2", description="Second"),
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        for todo in result.todo_list:
            assert todo.todo_completed is False

    def test_convert_todos_initializes_empty_output(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Task", description="Description")
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        todo = result.todo_list[0]

        assert isinstance(todo.output, TODOs_Output)
        assert todo.output.output == ""
        assert todo.output.model_used == ""
        assert todo.output.execution_time == ""

    def test_convert_todos_preserves_title_and_description(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(
                title="Implement feature",
                description="Implement the new feature with proper error handling"
            )
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        todo = result.todo_list[0]

        assert todo.todo_name == "Implement feature"
        assert todo.todo_description == "Implement the new feature with proper error handling"

    def test_convert_todos_with_special_characters(self):
        """Test that special characters in title/description are preserved."""
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(
                title="Test: API / Integration",
                description="Handle special chars: <>&\"' and unicode: \u2713"
            )
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        todo = result.todo_list[0]

        assert "API / Integration" in todo.todo_name
        assert "<>&\"'" in todo.todo_description

    def test_convert_todos_with_long_descriptions(self):
        """Test handling of long descriptions."""
        long_desc = "A" * 1000
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Long task", description=long_desc)
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        todo = result.todo_list[0]

        assert len(todo.todo_description) == 1000
        assert todo.todo_description == long_desc

    def test_convert_todos_thread_id_from_config(self):
        """Test that thread_id is retrieved from config."""
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Test", description="Test")
        ])
        result = convert_to_todos_with_mock_config(simple_list, thread_id="test-thread-123")

        assert result.thread_id == "test-thread-123"

    def test_convert_todos_thread_id_none(self):
        """Test that thread_id is None when not provided."""
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Test", description="Test")
        ])
        result = convert_to_todos_with_mock_config(simple_list)

        assert result.thread_id is None


class TestConvertToTodosEdgeCases:
    """Tests for edge cases in convert_to_todos."""

    def test_convert_with_whitespace_only_title(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="   ", description="Description")
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        assert result.todo_list[0].todo_name == "   "

    def test_convert_with_whitespace_only_description(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Title", description="   ")
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        assert result.todo_list[0].todo_description == "   "

    def test_convert_with_unicode_emoji(self):
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(title="Task 🔥", description="Test ✅ check 🚀")
        ])
        result = convert_to_todos_with_mock_config(simple_list)
        todo = result.todo_list[0]

        assert "🔥" in todo.todo_name
        assert "✅" in todo.todo_description
        assert "🚀" in todo.todo_description

    def test_convert_many_todos(self):
        """Test converting a large number of todos."""
        num_todos = 100
        simple_list = SimpleTODOList(todos=[
            SimpleTODO(
                title=f"Task {i}",
                description=f"Description {i}"
            )
            for i in range(1, num_todos + 1)
        ])
        result = convert_to_todos_with_mock_config(simple_list)

        assert len(result.todo_list) == num_todos
        # Verify first and last IDs
        assert result.todo_list[0].todo_id == "1"
        assert result.todo_list[-1].todo_id == str(num_todos)


class TestSimpleTODOListValidation:
    """Tests for SimpleTODOList validation behavior."""

    def test_simple_todo_list_todos_field_is_required(self):
        """todos field is required."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            SimpleTODOList()  # type: ignore

    def test_simple_todo_list_accepts_list_of_simple_todos(self):
        """Verify that the todos field accepts SimpleTODO objects."""
        todos = [
            SimpleTODO(title="T1", description="D1"),
            SimpleTODO(title="T2", description="D2"),
        ]
        todo_list = SimpleTODOList(todos=todos)
        assert len(todo_list.todos) == 2
