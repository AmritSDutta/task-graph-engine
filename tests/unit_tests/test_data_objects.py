"""Unit tests for data objects - Pydantic models."""

import pytest

from task_agent.data_objs.task_details import (
    TODO_details,
    TODOs,
    TODOs_Output,
)


class TestTODOsOutput:
    """Tests for TODOs_Output model."""

    def test_create_empty_output(self):
        output = TODOs_Output()
        assert output.output == ""
        assert output.model_used == ""
        assert output.execution_time == ""

    def test_create_output_with_values(self):
        output = TODOs_Output(
            output="Task completed successfully",
            model_used="gpt-4o",
            execution_time="1.23s"
        )
        assert output.output == "Task completed successfully"
        assert output.model_used == "gpt-4o"
        assert output.execution_time == "1.23s"

    def test_output_fields_are_mutable(self):
        output = TODOs_Output()
        output.output = "New result"
        output.model_used = "gemini-2.5-flash"
        output.execution_time = "0.5s"
        assert output.output == "New result"
        assert output.model_used == "gemini-2.5-flash"
        assert output.execution_time == "0.5s"


class TestTODODetails:
    """Tests for TODO_details model."""

    def test_create_todo_details_with_required_fields(self):
        todo = TODO_details(
            todo_name="Write unit tests",
            todo_description="Write pytest tests for all modules"
        )
        assert todo.todo_name == "Write unit tests"
        assert todo.todo_description == "Write pytest tests for all modules"
        assert todo.todo_id is None
        assert todo.todo_completed is False
        assert todo.output is None

    def test_create_todo_details_with_all_fields(self):
        output = TODOs_Output(output="Done", model_used="gpt-4o")
        todo = TODO_details(
            todo_id="todo-1",
            todo_name="Write docs",
            todo_description="Write comprehensive documentation",
            todo_completed=True,
            output=output
        )
        assert todo.todo_id == "todo-1"
        assert todo.todo_name == "Write docs"
        assert todo.todo_description == "Write comprehensive documentation"
        assert todo.todo_completed is True
        assert todo.output == output

    def test_todo_completed_defaults_to_false(self):
        todo = TODO_details(
            todo_name="Test",
            todo_description="Test description"
        )
        assert todo.todo_completed is False

    def test_todo_id_can_be_string(self):
        todo = TODO_details(
            todo_id="abc-123",
            todo_name="Test",
            todo_description="Test"
        )
        assert todo.todo_id == "abc-123"


class TestTODOs:
    """Tests for TODOs container model."""

    def test_create_empty_todos(self):
        todos = TODOs(todo_list=[])
        assert todos.todo_list == []
        assert todos.thread_id is None

    def test_create_todos_with_list(self):
        todo1 = TODO_details(
            todo_id="1",
            todo_name="Task 1",
            todo_description="First task"
        )
        todo2 = TODO_details(
            todo_id="2",
            todo_name="Task 2",
            todo_description="Second task"
        )
        todos = TODOs(todo_list=[todo1, todo2])
        assert len(todos.todo_list) == 2
        assert todos.todo_list[0].todo_name == "Task 1"
        assert todos.todo_list[1].todo_name == "Task 2"

    def test_create_todos_with_thread_id(self):
        todo = TODO_details(
            todo_name="Test",
            todo_description="Test"
        )
        todos = TODOs(todo_list=[todo], thread_id="thread-abc-123")
        assert todos.thread_id == "thread-abc-123"

    def test_todos_list_is_mutable(self):
        todos = TODOs(todo_list=[])
        new_todo = TODO_details(
            todo_name="New task",
            todo_description="New description"
        )
        todos.todo_list.append(new_todo)
        assert len(todos.todo_list) == 1


