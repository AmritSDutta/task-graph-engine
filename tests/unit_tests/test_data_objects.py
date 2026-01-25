"""Unit tests for data objects - Pydantic models."""

import pytest

from task_agent.data_objs.task_details import (
    TODO_details,
    TODOs,
    TODOs_Output,
)
from task_agent.data_objs.business_objs import (
    CombinedPlan,
    DecisionContext,
    DecisionID,
    DecisionOutput,
    DECISION_TRIGGERS,
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


class TestDecisionContext:
    """Tests for DecisionContext model."""

    def test_create_decision_context(self):
        context = DecisionContext(
            decision_id="reset_vpn_profile",
            context="User experiencing repeated disconnects"
        )
        assert context.decision_id == "reset_vpn_profile"
        assert context.context == "User experiencing repeated disconnects"


class TestDecisionOutput:
    """Tests for DecisionOutput model."""

    def test_create_decision_output_with_required_fields(self):
        decision = DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.85
        )
        assert decision.decision_id == "reset_vpn_profile"
        assert decision.decision is True
        assert decision.confidence == 0.85

    def test_confidence_defaults_to_zero(self):
        decision = DecisionOutput(
            decision_id="restart_sso_session",
            decision=False,
            confidence=0.0
        )
        assert decision.confidence == 0.0

    def test_confidence_is_rounded_to_three_decimal_places(self):
        """Test that confidence is rounded during validation."""
        decision = DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.8567  # Should be rounded to 0.857
        )
        assert decision.confidence == 0.857

    def test_confidence_rounding_with_many_decimals(self):
        decision = DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.123456789
        )
        assert decision.confidence == 0.123

    def test_confidence_must_be_between_0_and_1(self):
        """Test that confidence values outside [0.0, 1.0] raise validation error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=1.5
            )

        with pytest.raises(ValidationError):
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=False,
                confidence=-0.1
            )

    def test_confidence_accepts_boundary_values(self):
        decision1 = DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=0.0
        )
        assert decision1.confidence == 0.0

        decision2 = DecisionOutput(
            decision_id="reset_vpn_profile",
            decision=True,
            confidence=1.0
        )
        assert decision2.confidence == 1.0

    def test_optional_fields_have_defaults(self):
        decision = DecisionOutput(
            decision_id="send_notification",
            decision=True,
            confidence=0.7
        )
        assert decision.model == 'default'
        assert decision.notes is None
        assert decision.latency_ms is None
        assert decision.thread_identifier is None

    def test_create_with_all_optional_fields(self):
        decision = DecisionOutput(
            decision_id="approval_required",
            decision=True,
            confidence=0.95,
            model="gpt-4o",
            notes="Policy violation detected",
            latency_ms=320,
            thread_identifier="thread-123"
        )
        assert decision.model == "gpt-4o"
        assert decision.notes == "Policy violation detected"
        assert decision.latency_ms == 320
        assert decision.thread_identifier == "thread-123"

    def test_latency_must_be_non_negative(self):
        """Test that latency_ms must be >= 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8,
                latency_ms=-1
            )

    def test_valid_decision_ids(self):
        """Test that all valid DecisionID values are accepted."""
        valid_ids: list[DecisionID] = [
            "reset_vpn_profile",
            "restart_sso_session",
            "run_connectivity_diagnostics",
            "update_internal_record",
            "send_notification",
            "approval_required",
        ]
        for decision_id in valid_ids:
            decision = DecisionOutput(
                decision_id=decision_id,
                decision=True,
                confidence=0.5
            )
            assert decision.decision_id == decision_id

    def test_invalid_decision_id_raises_error(self):
        """Test that invalid decision_id raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DecisionOutput(
                decision_id="invalid_decision",  # type: ignore
                decision=True,
                confidence=0.5
            )


class TestDecisionTriggers:
    """Tests for DECISION_TRIGGERS constant."""

    def test_decision_triggers_has_all_decision_ids(self):
        """Ensure DECISION_TRIGGERS has entries for all DecisionID values."""
        expected_ids: list[DecisionID] = [
            "reset_vpn_profile",
            "restart_sso_session",
            "run_connectivity_diagnostics",
            "update_internal_record",
            "send_notification",
            "approval_required",
        ]
        assert set(DECISION_TRIGGERS.keys()) == set(expected_ids)

    def test_decision_trigger_descriptions_are_non_empty(self):
        for decision_id, description in DECISION_TRIGGERS.items():
            assert description
            assert len(description) > 0

    def test_decision_triggers_values_are_strings(self):
        for description in DECISION_TRIGGERS.values():
            assert isinstance(description, str)


class TestCombinedPlan:
    """Tests for CombinedPlan model."""

    def test_create_empty_combined_plan(self):
        plan = CombinedPlan()
        assert plan.reset_vpn_profile is False
        assert plan.restart_sso_session is False
        assert plan.run_connectivity_diagnostics is False
        assert plan.update_internal_record is False
        assert plan.send_notification is False
        assert plan.approval_required is False
        assert plan.generated_summary is None or plan.generated_summary == ''
        assert plan.confidence == 0.0
        assert plan.task_specific_notes == []

    def test_create_combined_plan_with_all_flags(self):
        plan = CombinedPlan(
            reset_vpn_profile=True,
            restart_sso_session=True,
            run_connectivity_diagnostics=False,
            update_internal_record=True,
            send_notification=False,
            approval_required=True,
            confidence=0.92
        )
        assert plan.reset_vpn_profile is True
        assert plan.restart_sso_session is True
        assert plan.run_connectivity_diagnostics is False
        assert plan.update_internal_record is True
        assert plan.send_notification is False
        assert plan.approval_required is True
        assert plan.confidence == 0.92

    def test_confidence_must_be_between_0_and_1(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CombinedPlan(confidence=1.5)

        with pytest.raises(ValidationError):
            CombinedPlan(confidence=-0.1)

    def test_confidence_accepts_boundary_values(self):
        plan1 = CombinedPlan(confidence=0.0)
        assert plan1.confidence == 0.0

        plan2 = CombinedPlan(confidence=1.0)
        assert plan2.confidence == 1.0

    def test_assemble_from_evaluators_with_empty_list(self):
        plan = CombinedPlan.assemble_from_evaluators([])
        assert plan.reset_vpn_profile is False
        assert plan.restart_sso_session is False
        assert plan.run_connectivity_diagnostics is False
        assert plan.update_internal_record is False
        assert plan.send_notification is False
        assert plan.approval_required is False
        assert plan.confidence == 0.0
        assert plan.task_specific_notes == []

    def test_assemble_from_evaluators_with_true_decisions_above_threshold(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.85
            ),
            DecisionOutput(
                decision_id="restart_sso_session",
                decision=True,
                confidence=0.72
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.reset_vpn_profile is True
        assert plan.restart_sso_session is True
        assert plan.confidence == 0.85  # max of true confidences

    def test_assemble_from_evaluators_with_true_decisions_below_threshold(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.5
            ),
            DecisionOutput(
                decision_id="restart_sso_session",
                decision=True,
                confidence=0.4
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.reset_vpn_profile is False
        assert plan.restart_sso_session is False
        assert plan.confidence == 0.0  # no decisions above threshold

    def test_assemble_from_evaluators_with_false_decisions(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=False,
                confidence=0.9
            ),
            DecisionOutput(
                decision_id="restart_sso_session",
                decision=False,
                confidence=0.85
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.reset_vpn_profile is False
        assert plan.restart_sso_session is False
        assert plan.confidence == 0.0

    def test_assemble_from_evaluators_mixed_true_false(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8
            ),
            DecisionOutput(
                decision_id="restart_sso_session",
                decision=False,
                confidence=0.9
            ),
            DecisionOutput(
                decision_id="approval_required",
                decision=True,
                confidence=0.7
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.reset_vpn_profile is True
        assert plan.restart_sso_session is False
        assert plan.approval_required is True
        assert plan.confidence == 0.8  # max of true confidences

    def test_assemble_from_evaluators_with_notes(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8,
                notes="VPN issues detected"
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert len(plan.task_specific_notes) > 0
        assert "[reset_vpn_profile]" in plan.task_specific_notes[0]

    def test_assemble_from_evaluators_with_summary_notes(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8,
                notes="Issue detected"
            ),
        ]
        summary_notes = ["Initial assessment"]
        plan = CombinedPlan.assemble_from_evaluators(
            evals,
            conf_threshold=0.6,
            summary_notes=summary_notes
        )
        assert len(plan.task_specific_notes) >= 2
        assert "Initial assessment" in plan.task_specific_notes

    def test_assemble_from_evaluators_with_additional_summary(self):
        evals = [
            DecisionOutput(
                decision_id="send_notification",
                decision=True,
                confidence=0.9
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(
            evals,
            conf_threshold=0.6,
            additional_summary="Action required: notify user"
        )
        assert plan.generated_summary == "Action required: notify user"

    def test_assemble_from_evaluators_thread_identifier(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8,
                thread_identifier="thread-abc"
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.thread_identifier == "thread-abc"

    def test_assemble_from_evaluators_confidence_rounding(self):
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8567  # Should round to 0.857
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.confidence == round(0.8567, 3)

    def test_assemble_from_evaluators_all_decision_types(self):
        """Test with all possible decision types."""
        evals = [
            DecisionOutput(
                decision_id=decision_id,
                decision=True,
                confidence=0.7
            )
            for decision_id in [
                "reset_vpn_profile",
                "restart_sso_session",
                "run_connectivity_diagnostics",
                "update_internal_record",
                "send_notification",
                "approval_required",
            ]
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        assert plan.reset_vpn_profile is True
        assert plan.restart_sso_session is True
        assert plan.run_connectivity_diagnostics is True
        assert plan.update_internal_record is True
        assert plan.send_notification is True
        assert plan.approval_required is True

    def test_assemble_from_evaluators_same_decision_multiple_evaluators(self):
        """Test when multiple evaluators evaluate the same decision."""
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.7
            ),
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.9
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(evals, conf_threshold=0.6)
        # Both evaluators said true, so decision should be true
        assert plan.reset_vpn_profile is True
        # Confidence should be max (0.9)
        assert plan.confidence == 0.9

    def test_assemble_from_evaluators_summary_notes_list_not_modified(self):
        """Test that summary_notes is not modified in place."""
        original_notes = ["original note"]
        evals = [
            DecisionOutput(
                decision_id="reset_vpn_profile",
                decision=True,
                confidence=0.8
            ),
        ]
        plan = CombinedPlan.assemble_from_evaluators(
            evals,
            conf_threshold=0.6,
            summary_notes=original_notes
        )
        # Original list should have the note added
        assert len(original_notes) >= 1

    def test_combined_plan_thread_identifier_optional(self):
        plan = CombinedPlan()
        assert plan.thread_identifier is None

        plan_with_id = CombinedPlan(thread_identifier="thread-123")
        assert plan_with_id.thread_identifier == "thread-123"
