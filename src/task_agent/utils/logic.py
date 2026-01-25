import logging
from typing import List

from langchain_core.messages import BaseMessage, convert_to_messages, get_buffer_string, AIMessage
from langgraph.config import get_config
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from pydantic import BaseModel

from task_agent.utils.state import Context
from task_agent.llms.llm_model_factory.llm_factory import create_llm
from task_agent.llms.simple_llm_selector import get_cheapest_model
from task_agent.utils.state import TaskState
from task_agent.data_objs.task_details import TODOs, TODO_details, TODOs_Output


# Simple schema for LLM output
class SimpleTODO(BaseModel):
    title: str
    description: str


class SimpleTODOList(BaseModel):
    todos: List[SimpleTODO]


def convert_to_todos(simple_list: SimpleTODOList) -> TODOs:
    """Convert simple LLM output to full TODOs structure"""
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

    cfg = get_config()
    thread_id = cfg.get("configurable", {}).get("thread_id")
    return TODOs(todo_list=todo_details_list, thread_id=thread_id)


async def entry_node(state: TaskState):
    if state.get("ended_once"):
        # Mark as closed
        return {"ended_once": True, "messages": AIMessage('Use another thread for run. It is already ended')}

    # Initialize empty todos if not present
    if "todos" not in state or not state.get("todos"):
        cfg = get_config()
        thread_id = cfg.get("configurable", {}).get("thread_id")
        logging.info(f"thread_id: {thread_id}")
        todo_to_be_updated = TODOs(todo_list=[])
        todo_to_be_updated.thread_id = thread_id
        state["todos"] = todo_to_be_updated

    return state


async def should_continue(state: TaskState):
    """Conditional edge: check if closed"""
    if state.get("ended_once"):
        logging.info("Thread already closed, skipping execution")
        return END

    return "planner"  # Normal flow


async def call_planner_model(state: TaskState, runtime: Runtime[Context]) -> Command:
    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.info(gbt)
    if not user_message:
        logging.info(ctm)
        return Command(update={"retry_count": state["retry_count"], "messages": state["messages"]}, goto=END)

    cheapest = await get_cheapest_model(gbt)
    logging.info(f"Cheapest model: {cheapest}")

    # System prompt for structured TODO generation
    system_prompt = """You are a task planning assistant. Analyze the user's task and generate a structured TODO list.

        Your output must be JSON with this structure:
        {
          "todos": [
            {
              "title": "Short task title",
              "description": "Detailed description of what needs to be done"
            },
            {
              "title": "Another task",
              "description": "Another detailed description"
            }
          ]
        }
        
        Generate 3-7 meaningful TODO items based on the user's task.
    """

    llm = create_llm(cheapest, temperature=0.0)
    structured_llm = llm.with_structured_output(SimpleTODOList)

    # Combine system prompt with user input
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": gbt}
    ]

    simple_todos = await structured_llm.ainvoke(prompt)

    # Convert simple output to full TODOs structure
    todos_response = convert_to_todos(simple_todos)

    logging.info(f"Todos response: {todos_response}")

    return Command(update={
        "task": gbt[:100],
        "messages": AIMessage(f"Generated {len(todos_response.todo_list)} TODOs, Todos response: {todos_response}"),
        "todos": todos_response,
        "ended_once": True
    }, goto='END')


async def call_subtask_model(state: TaskState, runtime: Runtime[Context]):
    """
       This will be called for each decision type needed.
       Worker: evaluate a single decision_id and append DecisionOutput.
       Expects state["decision_id"] injected via Send().
    """
    cfg = get_config()
    _thread_id = cfg.get("configurable", {}).get("thread_id", '')

    todo: TODO_details = state["todo"]
    logging.info(f'[Sub-task] {todo.todo_name} executing ... ')

    # System prompt for structured
    system_prompt = """
    You are a helpful assistant. 
    Analyze the user's task and generate a appropriate response.
    """

    todo_formated = f"ID: {todo.todo_id}\nTitle: {todo.todo_name}\nDescription: {todo.todo_description}"

    cheapest = await get_cheapest_model(system_prompt)
    logging.info(f"Cheapest model: {cheapest}")

    llm = create_llm(cheapest, temperature=0.0)
    # Combine system prompt with user input
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": todo_formated}
    ]
    response: AIMessage = await llm.ainvoke(prompt)
    logging.info(f'[todo] {todo}: {response.content}, {response.response_metadata}')
    return {
        "messages": AIMessage(f'Evaluated needs : {todo}'),
        "completed_todos": [response.content]
    }


async def assign_workers(state: TaskState, runtime: Runtime[Context]):
    """Assign a worker to each type of decision"""
    todos = state["todos"].todo_list

    if not todos:
        return "__end__"

    return [
        Send("subtask", {**state, "todo": td})
        for td in todos
    ]


async def call_combiner_model(state: TaskState, runtime: Runtime[Context]) -> Command:
    issue_summary: str = state['task']
    completed_todos: list[str] = state['completed_todos']
    logging.info(f"Combiner received {len(completed_todos)} completed todos")
    # System prompt for structured TODO generation
    system_prompt = """
        You are a helpful synthesizer assistant. 
        Generate a synthesized response.
        """
    formatted_todos = "\n".join(f"{i + 1}. {todo}" for i, todo in enumerate(completed_todos))

    cheapest = await get_cheapest_model(system_prompt)
    logging.info(f"Cheapest model: {cheapest}")

    llm = create_llm(cheapest, temperature=0.0)
    # Combine system prompt with user input
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": formatted_todos}
    ]
    response: AIMessage = await llm.ainvoke(prompt)
    final_output: str = response.content

    if not final_output:
        return Command(update={"messages": state["messages"]}, goto=END)

    return Command(update={
        "ended_once": True,
        "final_report": final_output,
        "messages": response
    }, goto=END)
