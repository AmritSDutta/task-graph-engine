import operator
from operator import add
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict, NotRequired

from task_agent.data_objs.task_details import TODOs, TODO_details


class Context(TypedDict):
    my_configurable_param: str


class TaskState(TypedDict):
    thread_id: str
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    todos: TODOs
    todo: NotRequired[TODO_details]
    completed_todos: Annotated[
        list, operator.add
    ]
    final_report: str
    ended_once: bool
    retry_count: Annotated[int, add]
