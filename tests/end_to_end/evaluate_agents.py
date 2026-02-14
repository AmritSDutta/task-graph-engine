"""
Experiment for using agents instead of chatmodels for planning
"""
import asyncio
import logging
from typing import List

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain_core.tools import BaseTool
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch


async def _get_mcp_server_details():
    import json
    from pathlib import Path

    # Resolve path from project root (find pyproject.toml or langgraph.json)
    current_path = Path(__file__).resolve()
    project_root = current_path
    for _ in range(10):  # Max 10 levels up to prevent infinite loop
        if (project_root / "pyproject.toml").exists():
            break
        if (project_root / "langgraph.json").exists():
            break
        parent = project_root.parent
        if parent == project_root:  # Reached filesystem root
            break
        project_root = parent

    path = project_root / "mcp_details.json"
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # dict or list, depending on JSON
    return data


async def get_mcp_tools(mcp_server_details) -> List[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    server_details = await mcp_server_details
    client = MultiServerMCPClient(server_details)
    return await client.get_tools()


async def get_agent():
    from langchain.agents.middleware import TodoListMiddleware
    from langchain.agents.middleware import ModelFallbackMiddleware
    from langchain.agents.middleware import ToolCallLimitMiddleware
    from langchain.agents.middleware import ToolRetryMiddleware
    from task_agent.llms.prompts import get_planner_prompt

    import os
    model = ChatOllama(
        model='glm-5:cloud',  # "mistral-large-3:675b-cloud",
        base_url="https://ollama.com",  # Cloud endpoint
        client_kwargs={
            "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY")},
            "timeout": 60.0  # Timeout in seconds
        }
    )
    tools: List[BaseTool] = await get_mcp_tools(mcp_server_details=_get_mcp_server_details())
    tools.append(TavilySearch())

    agent = create_agent(
        model,
        tools=tools,
        middleware=[
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ModelFallbackMiddleware(
                "google_genai:gemini-3-flash-preview",
            ),
            ToolCallLimitMiddleware(thread_limit=10, run_limit=5),
            ToolCallLimitMiddleware(
                tool_name="TavilySearch",
                thread_limit=2,
                run_limit=1,
            ),
            ToolRetryMiddleware(
                max_retries=2,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            TodoListMiddleware(),

        ],
        system_prompt=get_planner_prompt()
    )

    return agent


async def evaluate_agents():
    from langchain_core.messages import HumanMessage

    agent = await get_agent()
    user_query = "why sky is blue ?"
    from langchain_core.callbacks import get_usage_metadata_callback
    from langchain_core.messages import HumanMessage

    # Wrap your agent invocation in the callback context manager
    with get_usage_metadata_callback() as cb:
        result = await agent.ainvoke({"messages": [HumanMessage(user_query)]})

        # Access consolidated usage metrics
        print(cb.usage_metadata)
    print(result)
    # Handle both dict and list responses
    messages = result.get("messages", []) if isinstance(result, dict) else result

    if not messages:
        # print("No messages in response")
        return

    for msg in messages:
        if hasattr(msg, "content") and msg.content:
            # print(f'message:\n{msg.content}\n')
            pass


async def call_chat():
    import os
    model = ChatOllama(
        model='glm-5:cloud',  # "mistral-large-3:675b-cloud",
        base_url="https://ollama.com",  # Cloud endpoint
        client_kwargs={
            "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY")},
            "timeout": 60.0  # Timeout in seconds
        }
    )
    from tavily import TavilyClient
    tavily = TavilyClient()
    tool_call_result = None
    try:
        tool_call_result = tavily.search(query='why sky is blue ?', max_results=10)

    except Exception as e:
        logging.error(f"[Planner] Error calling tavily: {e}")

    # Debug: Print the actual structure
    print(f"Result type: {type(tool_call_result)}")

    # Access content from Tavily response (dict format)
    search_content = ""
    if tool_call_result:
        # Handle dict response
        if isinstance(tool_call_result, dict):
            # Option 1: Use the answer if available
            if tool_call_result.get('answer'):
                search_content = tool_call_result['answer']
            # Option 2: Combine all result snippets
            else:
                snippets = []
                for result in tool_call_result.get('results', []):
                    content = result.get('content', '')
                    if content:
                        snippets.append(f"- {result.get('title', 'No title')}: {content}")
                search_content = "\n\n".join(snippets)

        # Handle object response (for backward compatibility)
        else:
            if hasattr(tool_call_result, 'answer') and tool_call_result.answer:
                search_content = tool_call_result.answer
            elif hasattr(tool_call_result, 'results'):
                snippets = []
                for result in tool_call_result.results:
                    if hasattr(result, 'content'):
                        snippets.append(result.content)
                search_content = "\n\n".join(snippets)

    print(f"\n=== Search Content ===")
    print(search_content[:500] if search_content else "NO CONTENT FOUND")
    print(f"=== End Search Content ===\n")

    print('why sky is blue ?' + search_content)
    response = await model.ainvoke('why sky is blue ?' + search_content)
    print(response)
    return


if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('httpx').setLevel(logging.DEBUG)
    asyncio.run(call_chat())
