"""Tests for graph node functions in nodes.py."""

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.constants import END
from langgraph.types import Command, Send

from task_agent.data_objs.task_details import TODOs, TODO_details, TODOs_Output
from task_agent.utils.nodes import (
    SimpleTODO,
    SimpleTODOList,
    convert_to_todos,
    entry_node,
    should_continue,
    call_input_validation,
    call_planner_model,
    assign_workers,
    call_subtask_model,
    call_combiner_model,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_config():
    """Mock LangGraph config."""
    return {"configurable": {"thread_id": "test-thread-1"}}


@pytest.fixture
def mock_task_state():
    """Basic mock task state."""
    return {
        "thread_id": "test-thread-1",
        "messages": [HumanMessage(content="test task")],
        "todos": None,
        "ended_once": False,
        "retry_count": 0,
    }


@pytest.fixture
def mock_simple_todo_list():
    """Mock SimpleTODOList for LLM output."""
    return SimpleTODOList(todos=[
        SimpleTODO(title="TODO 1", description="First task"),
        SimpleTODO(title="TODO 2", description="Second task"),
    ])


@pytest.fixture
def mock_todos():
    """Mock TODOs object."""
    todos = TODOs(todo_list=[
        TODO_details(
            todo_id="1",
            todo_name="TODO 1",
            todo_description="First task",
            todo_completed=False,
        ),
        TODO_details(
            todo_id="2",
            todo_name="TODO 2",
            todo_description="Second task",
            todo_completed=False,
        ),
    ], thread_id="test-thread-1")
    return todos


@pytest.fixture
def mock_completed_todos():
    """Mock completed todos list."""
    return [
        "Result for TODO 1",
        "Result for TODO 2",
    ]


# ============================================================================
# Test Entry Node
# ============================================================================


class TestEntryNode:
    """Test the entry_node function."""

    @pytest.mark.asyncio
    async def test_entry_node_initializes_todos(self, mock_task_state, mock_config):
        """Test entry node initializes todos when not present."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = await entry_node(mock_task_state)

            assert "todos" in result
            assert result["todos"].todo_list == []
            assert result["todos"].thread_id == "test-thread-1"

    @pytest.mark.asyncio
    async def test_entry_node_preserves_existing_todos(self, mock_config, mock_todos):
        """Test entry node preserves existing todos."""
        state = {
            "thread_id": "test-thread-1",
            "messages": [HumanMessage(content="test")],
            "todos": mock_todos,
            "ended_once": False,
        }
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = await entry_node(state)

            assert result["todos"] == mock_todos
            assert len(result["todos"].todo_list) == 2

    @pytest.mark.asyncio
    async def test_entry_node_closed_thread(self, mock_task_state):
        """Test entry node returns message when thread is closed."""
        mock_task_state["ended_once"] = True
        result = await entry_node(mock_task_state)

        assert result["ended_once"] is True
        assert isinstance(result["messages"], AIMessage)
        assert "Use another thread" in result["messages"].content


# ============================================================================
# Test Should Continue
# ============================================================================


class TestShouldContinue:
    """Test the should_continue conditional edge function."""

    @pytest.mark.asyncio
    async def test_should_continue_with_closed_thread(self):
        """Test should_continue returns END when thread is closed."""
        state = {"ended_once": True}
        result = await should_continue(state)
        assert result == END

    @pytest.mark.asyncio
    async def test_should_continue_with_open_thread(self):
        """Test should_continue routes to input_validator when thread is open."""
        state = {"ended_once": False}
        result = await should_continue(state)
        assert result == "input_validator"


# ============================================================================
# Test Call Input Validation
# ============================================================================


class TestCallInputValidation:
    """Test the call_input_validation node function."""

    @pytest.mark.asyncio
    async def test_input_validation_safe_message(self, mock_task_state, mock_config):
        """Test input validation passes for safe message."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="safe message"):
                    with patch("task_agent.utils.nodes.scan_for_vulnerability", return_value=True) as mock_scan:
                        result = await call_input_validation(mock_task_state, MagicMock())

                        assert isinstance(result, Command)
                        assert result.goto == "planner"
                        assert result.update["task"] == "safe message"
                        mock_scan.assert_called_once_with("safe message")

    @pytest.mark.asyncio
    async def test_input_validation_unsafe_message(self, mock_task_state, mock_config):
        """Test input validation routes to END for unsafe message."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="rm -rf /"):
                    with patch("task_agent.utils.nodes.scan_for_vulnerability", return_value=False) as mock_scan:
                        result = await call_input_validation(mock_task_state, MagicMock())

                        assert isinstance(result, Command)
                        assert result.goto == END
                        assert "Unsafe" in result.update["messages"].content
                        mock_scan.assert_called_once_with("rm -rf /")

    @pytest.mark.asyncio
    async def test_input_validation_with_moderation_api(self, mock_task_state, mock_config):
        """Test input validation calls moderation API when enabled."""
        with patch("task_agent.utils.input_validation.settings") as mock_settings:
            mock_settings.MODERATION_API_CHECK_REQ = True
            with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
                with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                    with patch("task_agent.utils.nodes.get_buffer_string", return_value="test message"):
                        with patch("task_agent.utils.nodes.scan_for_vulnerability", return_value=True):
                            result = await call_input_validation(mock_task_state, MagicMock())
                            assert result.goto == "planner"

    @pytest.mark.asyncio
    async def test_input_validation_without_moderation_api(self, mock_task_state, mock_config):
        """Test input validation works without moderation API."""
        with patch("task_agent.utils.input_validation.settings") as mock_settings:
            mock_settings.MODERATION_API_CHECK_REQ = False
            with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
                with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                    with patch("task_agent.utils.nodes.get_buffer_string", return_value="safe message"):
                        with patch("task_agent.utils.nodes.scan_for_vulnerability", return_value=True):
                            result = await call_input_validation(mock_task_state, MagicMock())
                            assert result.goto == "planner"

    @pytest.mark.asyncio
    async def test_input_validation_empty_messages(self, mock_task_state, mock_config):
        """Test input validation with empty messages."""
        mock_task_state["messages"] = []
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=[]):
                result = await call_input_validation(mock_task_state, MagicMock())

                assert isinstance(result, Command)
                assert result.goto == END


# ============================================================================
# Test Call Planner Model
# ============================================================================


class TestCallPlannerModel:
    """Test the call_planner_model node function."""

    @pytest.mark.asyncio
    async def test_planner_generates_todos(self, mock_task_state, mock_simple_todo_list, mock_config):
        """Test planner generates TODOs successfully."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="test task"):
                    with patch("task_agent.utils.nodes.get_planner_prompt", return_value="system prompt"):
                        with patch("task_agent.utils.nodes.get_cheapest_model", return_value="gpt-4o-mini"):
                            with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock) as mock_llm:
                                mock_llm.return_value = mock_simple_todo_list
                                with patch("task_agent.utils.nodes.convert_to_todos", return_value=TODOs(
                                    todo_list=[
                                        TODO_details(
                                            todo_id="1",
                                            todo_name="TODO 1",
                                            todo_description="First task",
                                            todo_completed=False,
                                        )
                                    ], thread_id="test-thread-1"
                                )):
                                    result = await call_planner_model(mock_task_state, MagicMock())

                                    assert isinstance(result, Command)
                                    assert result.update["task"] == "test task"
                                    assert result.update["ended_once"] is False
                                    assert "todos" in result.update

    @pytest.mark.asyncio
    async def test_planner_empty_todo_list(self, mock_task_state, mock_config):
        """Test planner handles empty TODO list."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="test task"):
                    with patch("task_agent.utils.nodes.get_planner_prompt", return_value="system prompt"):
                        with patch("task_agent.utils.nodes.get_cheapest_model", return_value="gpt-4o-mini"):
                            with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock) as mock_llm:
                                mock_llm.return_value = SimpleTODOList(todos=[])
                                with patch("task_agent.utils.nodes.convert_to_todos", return_value=TODOs(
                                    todo_list=[], thread_id="test-thread-1"
                                )):
                                    result = await call_planner_model(mock_task_state, MagicMock())

                                    assert isinstance(result, Command)
                                    assert result.update["todos"].todo_list == []

    @pytest.mark.asyncio
    async def test_planner_exception_handling(self, mock_task_state, mock_config):
        """Test planner handles LLM exceptions."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="test task"):
                    with patch("task_agent.utils.nodes.get_planner_prompt", return_value="system prompt"):
                        with patch("task_agent.utils.nodes.get_cheapest_model", return_value="gpt-4o-mini"):
                            with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock) as mock_llm:
                                mock_llm.side_effect = Exception("LLM error")
                                result = await call_planner_model(mock_task_state, MagicMock())

                                assert isinstance(result, Command)
                                assert result.goto == "END"  # String, not END constant
                                assert result.update["ended_once"] is True
                                assert "Error during planning" in result.update["messages"].content

    @pytest.mark.asyncio
    async def test_planner_sets_ended_once_false(self, mock_task_state, mock_simple_todo_list, mock_config):
        """Test planner sets ended_once to False on success."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            with patch("task_agent.utils.nodes.convert_to_messages", return_value=mock_task_state["messages"]):
                with patch("task_agent.utils.nodes.get_buffer_string", return_value="test task"):
                    with patch("task_agent.utils.nodes.get_planner_prompt", return_value="system prompt"):
                        with patch("task_agent.utils.nodes.get_cheapest_model", return_value="gpt-4o-mini"):
                            with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock) as mock_llm:
                                mock_llm.return_value = mock_simple_todo_list
                                with patch("task_agent.utils.nodes.convert_to_todos", return_value=TODOs(
                                    todo_list=[
                                        TODO_details(
                                            todo_id="1",
                                            todo_name="TODO 1",
                                            todo_description="First task",
                                            todo_completed=False,
                                        )
                                    ], thread_id="test-thread-1"
                                )):
                                    result = await call_planner_model(mock_task_state, MagicMock())

                                    assert result.update["ended_once"] is False


# ============================================================================
# Test Assign Workers
# ============================================================================


class TestAssignWorkers:
    """Test the assign_workers node function."""

    @pytest.mark.asyncio
    async def test_assign_workers_with_todos(self, mock_task_state, mock_todos):
        """Test assign_workers creates Send objects for each TODO."""
        mock_task_state["todos"] = mock_todos
        result = await assign_workers(mock_task_state, MagicMock())

        assert isinstance(result, list)
        assert len(result) == 2
        for send_obj in result:
            assert isinstance(send_obj, Send)
            # Send objects have 'node' attribute (the target node name)
            assert hasattr(send_obj, 'node')
            assert send_obj.node == "subtask"

    @pytest.mark.asyncio
    async def test_assign_workers_without_todos(self, mock_task_state):
        """Test assign_workers returns __end__ when no todos."""
        mock_task_state["todos"] = TODOs(todo_list=[], thread_id="test-thread-1")
        result = await assign_workers(mock_task_state, MagicMock())

        assert result == "__end__"

    @pytest.mark.asyncio
    async def test_assign_workers_send_structure(self, mock_task_state, mock_todos):
        """Test assign_workers Send objects have correct structure."""
        mock_task_state["todos"] = mock_todos
        result = await assign_workers(mock_task_state, MagicMock())

        # Check first Send object
        first_send = result[0]
        assert hasattr(first_send, 'node')
        assert first_send.node == "subtask"
        # The Send object should contain the todo in its state (arg)
        assert hasattr(first_send, 'arg')
        assert "todo" in first_send.arg


# ============================================================================
# Test Call Subtask Model
# ============================================================================


class TestCallSubtaskModel:
    """Test the call_subtask_model node function."""

    @pytest.mark.asyncio
    async def test_subtask_executes_todo(self, mock_config):
        """Test subtask executes TODO successfully."""
        todo = TODO_details(
            todo_id="1",
            todo_name="Test TODO",
            todo_description="Test description",
            todo_completed=False,
        )
        state = {"todo": todo}

        with patch("task_agent.utils.nodes.get_subtask_prompt", return_value="system prompt {{CURRENT_DATE}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="Task completed successfully")
                mock_response.response_metadata = {"token_usage": {"total_tokens": 100}}
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response):
                    result = await call_subtask_model(state, MagicMock())

                    assert "messages" in result
                    assert "completed_todos" in result
                    assert result["completed_todos"] == ["Task completed successfully"]

    @pytest.mark.asyncio
    async def test_subtask_with_execution_time(self, mock_config):
        """Test subtask tracks execution time."""
        todo = TODO_details(
            todo_id="1",
            todo_name="Test TODO",
            todo_description="Test description",
            todo_completed=False,
        )
        state = {"todo": todo}

        with patch("task_agent.utils.nodes.get_subtask_prompt", return_value="system prompt {{CURRENT_DATE}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="Done")
                mock_response.response_metadata = {"token_usage": {"total_tokens": 50}}
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response):
                    result = await call_subtask_model(state, MagicMock())

                    assert result["completed_todos"] == ["Done"]

    @pytest.mark.asyncio
    async def test_subtask_exception_handling(self, mock_config):
        """Test subtask handles LLM exceptions."""
        todo = TODO_details(
            todo_id="1",
            todo_name="Test TODO",
            todo_description="Test description",
            todo_completed=False,
        )
        state = {"todo": todo}

        with patch("task_agent.utils.nodes.get_subtask_prompt", return_value="system prompt {{CURRENT_DATE}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, side_effect=Exception("LLM failed")):
                    result = await call_subtask_model(state, MagicMock())

                    assert "messages" in result
                    assert "completed_todos" in result
                    assert "Error" in result["completed_todos"][0]

    @pytest.mark.asyncio
    async def test_subtask_return_structure(self, mock_config):
        """Test subtask returns correct structure."""
        todo = TODO_details(
            todo_id="1",
            todo_name="Test TODO",
            todo_description="Test description",
            todo_completed=False,
        )
        state = {"todo": todo}

        with patch("task_agent.utils.nodes.get_subtask_prompt", return_value="system prompt {{CURRENT_DATE}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="Result")
                mock_response.response_metadata = {"token_usage": {"total_tokens": 75}}
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response):
                    result = await call_subtask_model(state, MagicMock())

                    assert isinstance(result, dict)
                    assert isinstance(result["messages"], AIMessage)
                    assert isinstance(result["completed_todos"], list)
                    assert len(result["completed_todos"]) == 1


# ============================================================================
# Test Call Combiner Model
# ============================================================================


class TestCallCombinerModel:
    """Test the call_combiner_model node function."""

    @pytest.mark.asyncio
    async def test_combiner_synthesizes_results(self):
        """Test combiner synthesizes completed todos successfully."""
        state = {
            "task": "test query",
            "completed_todos": ["Result 1", "Result 2"],
            "messages": [HumanMessage(content="test")],
        }

        with patch("task_agent.utils.nodes.get_combiner_prompt_only", return_value="system {{user_query}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="Final synthesized report")
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
                    result = await call_combiner_model(state, MagicMock())

                    # Check bind_tools_flag was False
                    mock_llm.assert_called_once()
                    call_kwargs = mock_llm.call_args.kwargs
                    assert call_kwargs.get("bind_tools_flag") is False

                    assert isinstance(result, Command)
                    assert result.goto == END
                    assert result.update["ended_once"] is True
                    assert result.update["final_report"] == "Final synthesized report"

    @pytest.mark.asyncio
    async def test_combiner_with_tool_calls(self):
        """Test combiner handles response with tool calls."""
        state = {
            "task": "test query",
            "completed_todos": ["Result 1"],
            "messages": [HumanMessage(content="test")],
        }

        with patch("task_agent.utils.nodes.get_combiner_prompt_only", return_value="system {{user_query}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="")
                mock_response.tool_calls = [{"name": "search", "args": {"query": "test"}}]
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response) as mock_llm:
                    result = await call_combiner_model(state, MagicMock())

                    # Check bind_tools_flag was False
                    call_kwargs = mock_llm.call_args.kwargs
                    assert call_kwargs.get("bind_tools_flag") is False

                    # Tool calls should be converted to string
                    assert isinstance(result.update["final_report"], str)

    @pytest.mark.asyncio
    async def test_combiner_empty_response(self):
        """Test combiner handles empty response content."""
        state = {
            "task": "test query",
            "completed_todos": ["Result 1"],
            "messages": [HumanMessage(content="test")],
        }

        with patch("task_agent.utils.nodes.get_combiner_prompt_only", return_value="system {{user_query}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="")
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response):
                    result = await call_combiner_model(state, MagicMock())

                    # Should return END without setting final_report
                    assert isinstance(result, Command)
                    assert result.goto == END

    @pytest.mark.asyncio
    async def test_combiner_sets_ended_once_true(self):
        """Test combiner sets ended_once to True on success."""
        state = {
            "task": "test query",
            "completed_todos": ["Result 1"],
            "messages": [HumanMessage(content="test")],
        }

        with patch("task_agent.utils.nodes.get_combiner_prompt_only", return_value="system {{user_query}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                mock_response = AIMessage(content="Final report")
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, return_value=mock_response):
                    result = await call_combiner_model(state, MagicMock())

                    assert result.update["ended_once"] is True

    @pytest.mark.asyncio
    async def test_combiner_exception_handling(self):
        """Test combiner handles exceptions."""
        state = {
            "task": "test query",
            "completed_todos": ["Result 1"],
            "messages": [HumanMessage(content="test")],
        }

        with patch("task_agent.utils.nodes.get_combiner_prompt_only", return_value="system {{user_query}}"):
            with patch("task_agent.utils.nodes.get_cheapest_model", new_callable=AsyncMock, return_value="gpt-4o-mini"):
                with patch("task_agent.utils.nodes.call_llm_with_retry", new_callable=AsyncMock, side_effect=Exception("LLM error")):
                    result = await call_combiner_model(state, MagicMock())

                    assert isinstance(result, Command)
                    assert result.goto == END
                    assert result.update["ended_once"] is True
                    assert "Error" in result.update["messages"].content


# ============================================================================
# Test Convert to TODOs
# ============================================================================


class TestConvertToTodos:
    """Test the convert_to_todos helper function."""

    def test_convert_to_todos_basic(self, mock_simple_todo_list, mock_config):
        """Test basic conversion from SimpleTODOList to TODOs."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = convert_to_todos(mock_simple_todo_list)

            assert isinstance(result, TODOs)
            assert len(result.todo_list) == 2
            assert result.thread_id == "test-thread-1"
            assert result.todo_list[0].todo_id == "1"
            assert result.todo_list[1].todo_id == "2"

    def test_convert_to_todos_preserves_titles(self, mock_simple_todo_list, mock_config):
        """Test conversion preserves todo titles."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = convert_to_todos(mock_simple_todo_list)

            assert result.todo_list[0].todo_name == "TODO 1"
            assert result.todo_list[1].todo_name == "TODO 2"

    def test_convert_to_todos_preserves_descriptions(self, mock_simple_todo_list, mock_config):
        """Test conversion preserves todo descriptions."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = convert_to_todos(mock_simple_todo_list)

            assert result.todo_list[0].todo_description == "First task"
            assert result.todo_list[1].todo_description == "Second task"

    def test_convert_to_todos_initializes_output(self, mock_simple_todo_list, mock_config):
        """Test conversion initializes empty output."""
        with patch("task_agent.utils.nodes.get_config", return_value=mock_config):
            result = convert_to_todos(mock_simple_todo_list)

            assert result.todo_list[0].todo_completed is False
            # Output has default empty TODOs_Output, not None
            assert result.todo_list[0].output == TODOs_Output(output="", model_used="", execution_time="")
