import logging
from typing import List

from langchain_core.messages import BaseMessage, convert_to_messages, get_buffer_string, AIMessage
from langgraph.config import get_config
from langgraph.constants import END
from langgraph.runtime import Runtime
from langgraph.types import Command, Send
from pydantic import BaseModel

from task_agent.data_objs.task_details import TODOs, TODO_details, TODOs_Output
from task_agent.llms.simple_llm_selector import get_cheapest_model
from task_agent.utils.circuit_breaker import call_llm_with_retry
from task_agent.utils.input_validation import scan_for_vulnerability
from task_agent.utils.state import Context
from task_agent.utils.state import TaskState


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
        logging.info(f"[{thread_id}] Entry node - Initializing todos")
        todo_to_be_updated = TODOs(todo_list=[])
        todo_to_be_updated.thread_id = thread_id
        state["todos"] = todo_to_be_updated

    return state


async def should_continue(state: TaskState):
    """Conditional edge: check if closed"""
    if state.get("ended_once"):
        logging.info("Thread already closed, skipping execution")
        return END

    return "input_validator"  # Normal flow


async def call_planner_model(state: TaskState, runtime: Runtime[Context]) -> Command:
    cfg = get_config()
    thread_id = cfg.get("configurable", {}).get("thread_id")

    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.debug(f'[{thread_id}]raw request received: {gbt}')
    if not user_message:
        logging.info(ctm)
        return Command(update={"retry_count": state["retry_count"], "messages": state["messages"]}, goto=END)

    system_prompt = """
    You are a task planning assistant. Analyze the user's task 
    and generate a structured TODO list.
    You must perform all internal reasoning, task planning, and intermediate steps in English only.

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
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": gbt}
    ]

    cheapest = await get_cheapest_model(str(prompt))
    logging.info(f"[Planner] Model: {cheapest}")

    try:
        simple_todos = await call_llm_with_retry(
            cheapest,
            prompt,
            fallback_model="gpt-4o",  # Fallback to GPT-4o if cheapest fails
            structured_output=SimpleTODOList,
            temperature=0.0
        )
    except Exception as e:
        logging.error(f"[Planner] Error calling LLM: {e}")
        return Command(update={
            "task": gbt,
            "messages": AIMessage(f"Error during planning: {e}"),
            "ended_once": True
        }, goto='END')

    # Convert simple output to full TODOs structure
    todos_response: TODOs = convert_to_todos(simple_todos)

    logging.info(f"[PLANNER] Total TODOs: {len(todos_response.todo_list)}")
    return Command(update={
        "task": gbt,
        "messages": AIMessage(f"Generated {len(todos_response.todo_list)} TODOs for task: {gbt[:100]}"),
        "todos": todos_response,
        "ended_once": False
    })


async def call_subtask_model(state: TaskState, runtime: Runtime[Context]):
    """
       This will be called for each decision type needed.
       Worker: evaluate a single decision_id and append DecisionOutput.
       Expects state["decision_id"] injected via Send().
    """
    import time
    start = time.time()

    todo: TODO_details = state["todo"]
    logging.info(f"[{todo.todo_id}] STARTING: {todo.todo_name}")

    # System prompt for structured
    system_prompt = """
    You are a helpful assistant. 
    Analyze the user's task and generate a appropriate response.
    you can use tools if it is available.
    """

    todo_formated = f"ID: {todo.todo_id}\nTitle: {todo.todo_name}\nDescription: {todo.todo_description}"

    cheapest = await get_cheapest_model(system_prompt)
    logging.info(f"[{todo.todo_id}] Model: {cheapest}")

    # Combine system prompt with user input
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": todo_formated}
    ]
    try:
        response: AIMessage = await call_llm_with_retry(
            cheapest,
            prompt,
            fallback_model="gpt-4o",
            temperature=0.0
        )
    except Exception as e:
        logging.error(f"[{todo.todo_id}] Error calling LLM: {e}")
        return {
            "messages": AIMessage(f"[{todo.todo_id}] Error evaluating: {todo.todo_name} - {e}"),
            "completed_todos": [f"Error processing {todo.todo_id}: {e}"]
        }

    duration = time.time() - start
    logging.info({
        "event": "subtask_completed",
        "todo_id": todo.todo_id,
        "todo_name": todo.todo_name,
        "model": cheapest,
        "duration": duration,
        "tokens": response.response_metadata.get("token_usage")
    })
    return {
        "messages": AIMessage(f'[{todo.todo_id}] Evaluated: {todo.todo_name}'),
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
    import time
    start = time.time()
    issue_summary: str = state['task']
    completed_todos: list[str] = state['completed_todos']
    logging.info(f"Combiner received {len(completed_todos)} completed todos")
    system_prompt = """
        You are a helpful synthesizer assistant.                                                                                                                                                                                     
        The user's original request was: {user_query}                                                                                                                                                                                
                                                                                                                                                                                                                                   
        Generate a synthesized response in English.   
    """
    formatted_system_prompt = system_prompt.format(user_query=issue_summary)
    formatted_todos = "\n".join(
        f"{i + 1}. {todo[:500]}..." if len(todo) > 500 else f"{i + 1}. {todo}"
        for i, todo in enumerate(completed_todos)
    )

    cheapest = await get_cheapest_model(system_prompt)
    logging.info(f"[COMBINER] Model: {cheapest}")

    # Combine system prompt with user input
    prompt = [
        {"role": "system", "content": formatted_system_prompt},
        {"role": "user", "content": formatted_todos}
    ]
    final_output: str | None = None
    try:
        response: AIMessage = await call_llm_with_retry(
            cheapest,
            prompt,
            fallback_model="gpt-4o",
            temperature=0.0
        )
        final_output = response.content
    except Exception as e:
        logging.error(f"[COMBINER] Error calling LLM: {e}")
        return Command(update={
            "messages": AIMessage(f"Error during final report generation: {e}"),
            "ended_once": True
        }, goto=END)

    if not final_output:
        return Command(update={"messages": state["messages"]}, goto=END)
    duration = time.time() - start
    logging.info("=" * 60)
    logging.info(f"EXECUTION SUMMARY: {issue_summary[:100]}")
    logging.info(f"Total TODOs: {len(completed_todos)}")
    logging.info(f"Combiner execution time: {duration:.2f}s")
    logging.info("=" * 60)

    return Command(update={
        "ended_once": True,
        "final_report": final_output,
        "messages": response,
    }, goto=END)


async def call_input_validation(state: TaskState, runtime: Runtime[Context]) -> Command:
    cfg = get_config()
    thread_id = cfg.get("configurable", {}).get("thread_id")

    user_message: list[BaseMessage] = state.get("messages")
    ctm = convert_to_messages(user_message)
    gbt = get_buffer_string(ctm, human_prefix="", ai_prefix="").strip()
    logging.info(f'[{thread_id}] Input received for validation: {gbt[:100]}...')
    if not user_message:
        logging.info(ctm)
        return Command(update={"retry_count": state["retry_count"], "messages": state["messages"]}, goto=END)

    is_safe: bool = await scan_for_vulnerability(gbt)
    if is_safe:
        logging.info(f'[{thread_id}] Input validation passed')
        return Command(update={
            "task": gbt,
            "messages": AIMessage(f"Validated user prompt: {gbt[:50]}..."),
        }, goto="planner")
    else:
        logging.warning(f'[{thread_id}] Input validation failed - malicious content detected')
        return Command(update={"task": gbt, "messages": AIMessage(f"Unsafe user prompt detected: {gbt[:50]}...")},
                       goto=END)
