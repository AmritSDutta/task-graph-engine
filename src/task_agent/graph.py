from __future__ import annotations

import os

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from task_agent.config import settings
from task_agent.logging_config import setup_logging
from task_agent.utils.nodes import call_planner_model, entry_node, should_continue, call_subtask_model, \
    call_combiner_model, assign_workers, call_input_validation, route_after_validation, end_node
from task_agent.utils.state import Context, TaskState

setup_logging()
os.environ["LANGSMITH_TRACING_V2"] = settings.ENABLE_LANGSMITH_TRACING_V2

# this name is mentioned in langgraph.json
graph = (
    StateGraph(TaskState, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("input_validator", call_input_validation)
    .add_node("planner", call_planner_model)
    .add_node("subtask", call_subtask_model)
    .add_node("combiner", call_combiner_model)
    .add_node("end", end_node)
    .add_edge(START, "entry")
    .add_conditional_edges('entry', should_continue, {"input_validator": "input_validator", END: END})
    .add_conditional_edges("input_validator", route_after_validation, {"planner": "planner", END: END})
    .add_conditional_edges("planner", assign_workers)  # <- fan-out here
    .add_edge("subtask", "combiner")  # fan-in after workers
    .add_edge("combiner", "end")
    .add_edge("end", END)
)
