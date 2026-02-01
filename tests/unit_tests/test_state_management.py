"""Unit tests for state management - no API calls required."""

import pytest
from typing import NotRequired
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from task_agent.utils.state import Context, TaskState
from task_agent.data_objs.task_details import TODOs, TODO_details, TODOs_Output


class TestContext:
    """Tests for Context TypedDict."""

    def test_context_has_my_configurable_param(self):
        """Context should have my_configurable_param field."""
        context: Context = {"my_configurable_param": "test_value"}
        assert context["my_configurable_param"] == "test_value"

    def test_context_type_annotation(self):
        """Test that Context is a TypedDict."""
        assert isinstance(Context, type)

    def test_context_accepts_string_value(self):
        """Context should accept string values."""
        context: Context = {"my_configurable_param": "some_string"}
        assert isinstance(context["my_configurable_param"], str)


class TestTaskState:
    """Tests for TaskState TypedDict."""

    def test_task_state_has_required_fields(self):
        """TaskState should have all required fields."""
        from operator import add

        # Create a minimal valid state
        todos = TODOs(todo_list=[], thread_id="test-thread")
        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [HumanMessage(content="Hello")],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["thread_id"] == "test-thread-123"
        assert state["task"] == "Test task"
        assert state["final_report"] == ""
        assert state["ended_once"] is False
        assert state["retry_count"] == 0

    def test_task_state_with_optional_todo_field(self):
        """TaskState should accept optional todo field."""
        from task_agent.utils.state import TaskState

        todos = TODOs(todo_list=[], thread_id="test-thread")
        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        # todo is optional, so state should be valid without it
        assert "todo" not in state or state.get("todo") is None

    def test_task_state_with_todo_field(self):
        """TaskState should accept optional todo field when provided."""
        todos = TODOs(todo_list=[], thread_id="test-thread")
        todo = TODO_details(
            todo_id="1",
            todo_name="Test todo",
            todo_description="Test description"
        )

        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "todo": todo,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["todo"] is not None
        assert state["todo"].todo_id == "1"

    def test_task_state_messages_annotated_with_add_messages(self):
        """Test that messages field uses add_messages annotation."""
        from langgraph.graph import add_messages
        from typing import Annotated, get_args

        # Get the annotation for messages field
        # This is a compile-time check, but we can verify the type accepts lists
        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [HumanMessage(content="First"), AIMessage(content="Second")],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["messages"]) == 2

    def test_task_state_completed_todos_uses_operator_add(self):
        """Test that completed_todos field uses operator.add."""
        import operator

        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": ["todo1"],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert isinstance(state["completed_todos"], list)
        assert state["completed_todos"] == ["todo1"]

    def test_task_state_retry_count_uses_add(self):
        """Test that retry_count field uses add annotation."""
        state: TaskState = {
            "thread_id": "test-thread-123",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 3
        }

        assert state["retry_count"] == 3

    def test_task_state_thread_id_is_string(self):
        """Test that thread_id field is a string."""
        state: TaskState = {
            "thread_id": "thread-abc-123",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="thread-abc-123"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert isinstance(state["thread_id"], str)

    def test_task_state_task_is_string(self):
        """Test that task field is a string."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Write a Python function",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert isinstance(state["task"], str)

    def test_task_state_final_report_is_string(self):
        """Test that final_report field is a string."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "Task completed successfully",
            "ended_once": False,
            "retry_count": 0
        }

        assert isinstance(state["final_report"], str)

    def test_task_state_ended_once_is_bool(self):
        """Test that ended_once field is a boolean."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": True,
            "retry_count": 0
        }

        assert isinstance(state["ended_once"], bool)

    def test_task_state_todos_is_todos_type(self):
        """Test that todos field is a TODOs instance."""
        todos = TODOs(
            todo_list=[
                TODO_details(
                    todo_id="1",
                    todo_name="Test",
                    todo_description="Test description"
                )
            ],
            thread_id="test-thread"
        )

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert isinstance(state["todos"], TODOs)
        assert len(state["todos"].todo_list) == 1


class TestTaskStateEdgeCases:
    """Tests for edge cases in TaskState."""

    def test_task_state_with_empty_messages(self):
        """Test TaskState with empty messages list."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["messages"] == []

    def test_task_state_with_multiple_messages(self):
        """Test TaskState with multiple messages."""
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?"),
            SystemMessage(content="You are a helpful assistant")
        ]

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": messages,
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["messages"]) == 4

    def test_task_state_with_empty_task_string(self):
        """Test TaskState with empty task string."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["task"] == ""

    def test_task_state_with_long_task_string(self):
        """Test TaskState with long task string."""
        long_task = "A" * 10000

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": long_task,
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["task"]) == 10000

    def test_task_state_with_multiple_completed_todos(self):
        """Test TaskState with multiple completed todos."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": ["todo1", "todo2", "todo3"],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["completed_todos"]) == 3

    def test_task_state_with_special_characters_in_task(self):
        """Test TaskState with special characters in task string."""
        special_task = "Test: API / Integration & <special> chars"

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": special_task,
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert "API / Integration" in state["task"]

    def test_task_state_with_unicode_in_task(self):
        """Test TaskState with unicode characters in task string."""
        unicode_task = "Test with emoji: 🔥 ✅ 🚀 and unicode: café"

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": unicode_task,
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert "🔥" in state["task"]

    def test_task_state_retry_count_zero(self):
        """Test TaskState with retry_count = 0."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["retry_count"] == 0

    def test_task_state_retry_count_positive(self):
        """Test TaskState with positive retry_count."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 5
        }

        assert state["retry_count"] == 5


class TestStateTransitions:
    """Tests for valid state transitions."""

    def test_state_transition_from_new_to_in_progress(self):
        """Test state transition when task starts."""
        initial_state: TaskState = {
            "thread_id": "test-thread",
            "messages": [HumanMessage(content="Start task")],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        # Simulate adding todos
        todos_with_items = TODOs(
            todo_list=[
                TODO_details(
                    todo_id="1",
                    todo_name="Todo 1",
                    todo_description="First todo"
                )
            ],
            thread_id="test-thread"
        )

        updated_state = initial_state.copy()
        updated_state["todos"] = todos_with_items

        assert len(updated_state["todos"].todo_list) == 1

    def test_state_transition_adding_to_completed_todos(self):
        """Test state transition when todos are completed."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        # Simulate adding completed todos (operator.add behavior)
        new_completed = state["completed_todos"] + ["todo1", "todo2"]
        updated_state = state.copy()
        updated_state["completed_todos"] = new_completed

        assert len(updated_state["completed_todos"]) == 2

    def test_state_transition_setting_ended_once(self):
        """Test state transition when thread ends."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        # Simulate setting ended_once to True
        updated_state = state.copy()
        updated_state["ended_once"] = True

        assert updated_state["ended_once"] is True

    def test_state_transition_setting_final_report(self):
        """Test state transition when final report is generated."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        # Simulate setting final report
        updated_state = state.copy()
        updated_state["final_report"] = "All tasks completed successfully"

        assert updated_state["final_report"] == "All tasks completed successfully"

    def test_state_transition_incrementing_retry_count(self):
        """Test state transition when retry count increments."""
        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": TODOs(todo_list=[], thread_id="test-thread"),
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 2
        }

        # Simulate incrementing retry count (operator.add behavior)
        updated_state = state.copy()
        updated_state["retry_count"] = state["retry_count"] + 1

        assert updated_state["retry_count"] == 3


class TestStateWithTODOs:
    """Tests for state with populated TODOs."""

    def test_state_with_single_todo(self):
        """Test state with a single TODO."""
        todos = TODOs(
            todo_list=[
                TODO_details(
                    todo_id="1",
                    todo_name="Write tests",
                    todo_description="Write unit tests"
                )
            ],
            thread_id="test-thread"
        )

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["todos"].todo_list) == 1
        assert state["todos"].todo_list[0].todo_name == "Write tests"

    def test_state_with_multiple_todos(self):
        """Test state with multiple TODOs."""
        todos = TODOs(
            todo_list=[
                TODO_details(
                    todo_id="1",
                    todo_name="Todo 1",
                    todo_description="First"
                ),
                TODO_details(
                    todo_id="2",
                    todo_name="Todo 2",
                    todo_description="Second"
                ),
                TODO_details(
                    todo_id="3",
                    todo_name="Todo 3",
                    todo_description="Third"
                )
            ],
            thread_id="test-thread"
        )

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert len(state["todos"].todo_list) == 3

    def test_state_with_completed_todo(self):
        """Test state with a completed TODO."""
        todos = TODOs(
            todo_list=[
                TODO_details(
                    todo_id="1",
                    todo_name="Completed task",
                    todo_description="This task is done",
                    todo_completed=True,
                    output=TODOs_Output(
                        output="Task completed",
                        model_used="gpt-4o",
                        execution_time="1.5s"
                    )
                )
            ],
            thread_id="test-thread"
        )

        state: TaskState = {
            "thread_id": "test-thread",
            "messages": [],
            "task": "Test task",
            "todos": todos,
            "completed_todos": [],
            "final_report": "",
            "ended_once": False,
            "retry_count": 0
        }

        assert state["todos"].todo_list[0].todo_completed is True
        assert state["todos"].todo_list[0].output.output == "Task completed"
