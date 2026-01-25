from __future__ import annotations

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from task_agent.logging_config import setup_logging
from task_agent.utils.logic import call_planner_model, entry_node, should_continue
from task_agent.utils.state import Context, TaskState

setup_logging()


# this name is mentioned in langgraph.json
graph = (
    StateGraph(TaskState, context_schema=Context)
    .add_node("entry", entry_node)
    .add_node("planner", call_planner_model)
    .add_edge(START, "entry")
    .add_conditional_edges('entry', should_continue, {"planner": "planner", END: END})
    .add_edge("planner", END)
)
