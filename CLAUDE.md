# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A **LangGraph-based task planning system** that uses intelligent LLM selection to process user tasks. The system dynamically selects the most appropriate model based on task requirements and cost optimization using a factory pattern for multi-provider model creation.

**Tech Stack**: LangGraph, LangChain, OpenAI Agents SDK, Google GenAI, Groq, Ollama, Tavily (web search), Pydantic, pytest

**Architecture Pattern**: Fan-out/fan-in parallel task execution with capability-based model inference, exponential cost penalty for load balancing, and circuit breaker retry logic.

## Development Commands

### Project Installation
```bash
# Install package in editable mode (required for imports to work)
pip install -e .
```

**Important**: The package must be installed with `pip install -e .` before running `langgraph dev` or any code that imports `task_agent` modules. This is because the code uses absolute imports like `from task_agent.xxx` which require the package to be installed.

### Code Quality
```bash
ruff check .           # Run linting
ruff check --fix .     # Auto-fix linting issues
mypy src/              # Run type checking
```

### Testing
```bash
pytest                              # Run all tests
pytest tests/unit_tests/            # Run unit tests only
pytest tests/unit_tests/test_input_validation.py -v   # Run single test file
pytest tests/unit_tests/test_input_validation.py::TestScanForVulnerability::test_safe_message -v  # Run single test
```

**Test Configuration**:
- `pytest-asyncio` with `asyncio_mode = "auto"` and `asyncio_default_fixture_loop_scope = "session"`
- Auto-use fixture in `tests/conftest.py` sets test API keys via monkeypatch for isolation
- Test API keys: `OPENAI_API_KEY=test-openai-key-for-pytest`, `GOOGLE_API_KEY=test-google-key-for-pytest`
- Unit tests mock external API calls to avoid real network requests

### LangGraph Development
```bash
langgraph dev               # Start LangGraph development server
langgraph dev --allow-blocking  # Use for community models (z.ai, nvidia, etc.)
```

The server will start at http://127.0.0.1:2024 with Studio UI at https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

**Important**: Use `--allow-blocking` flag when using LangChain community integrations (like `langchain-nvidia-ai-core`, `langchain-community`, `z.ai`) that use synchronous HTTP calls internally. This prevents LangGraph from throwing warnings about blocking calls in an async context. Standard providers (OpenAI, Google/Gemini, Groq, Anthropic) work fine without this flag.

**Testing the API**:
```bash
# Send a test request to LangGraph core endpoint (mounted at /langgraph)
curl -X POST http://127.0.0.1:2024/langgraph/runs \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "agent", "input": {"messages": [{"role": "user", "content": "your task here"}]}}'

# Test health check endpoint
curl http://127.0.0.1:2024/api/health
```

Note: Runs with `temporary: true` (default) are cleaned up after completion and will return 404 when queried later.

### Custom REST API

The server includes a custom FastAPI app (`webapp.py`) that provides additional REST endpoints alongside the LangGraph core API.

**Base URL**: `http://127.0.0.1:2024`

**Available Endpoints**:

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Root endpoint with API information | No |
| GET | `/api/health` | Health check and system info | No |
| GET | `/docs` | Interactive OpenAPI docs (Swagger UI) | No |
| GET | `/redoc` | Alternative API documentation (ReDoc) | No |
| GET | `/api/models` | List all available models with capabilities and costs | If enabled |
| GET | `/api/models/{model_name}` | Get details for a specific model | If enabled |
| GET | `/api/statistics` | Get runtime statistics (model usage, costs) | If enabled |
| GET | `/api/config` | Get current configuration settings | If enabled |
| * | `/langgraph/*` | LangGraph core API (runs, threads, etc.) | No |

**Authentication**:

Protected endpoints require API key authentication via Bearer token when `REQUIRE_AUTH=true`:

```bash
# Set environment variables (in .env file or directly)
API_KEY=your-secret-api-key-here
REQUIRE_AUTH=true

# Test protected endpoint
curl -H "Authorization: Bearer your-secret-api-key-here" \
  http://127.0.0.1:2024/api/models
```

**Example Responses**:

```bash
# Health check
curl http://127.0.0.1:2024/api/health
# {"status":"healthy","version":"1.0.0","auth_required":false,"models_loaded":15}

# List models (with auth if enabled)
curl -H "Authorization: Bearer your-key" http://127.0.0.1:2024/api/models | jq
# {
#   "count": 15,
#   "models": {
#     "gpt-4o-mini": {
#       "capabilities": ["reasoning", "tools", "fast", "cheap", "informational", "coding"],
#       "cost": 0.035,
#       "is_coding_priority": false
#     },
#     ...
#   }
# }

# Get statistics (with auth if enabled)
curl -H "Authorization: Bearer your-key" http://127.0.0.1:2024/api/statistics | jq
# {
#   "cost_spreading_factor": 0.03,
#   "total_models_with_usage": 3,
#   "models": {
#     "gpt-4o-mini": {
#       "usage_count": 5,
#       "base_cost": 0.035,
#       "derived_cost": 0.0405,
#       "penalty_factor": 1.1576
#     }
#   }
# }
```

## Architecture

### Graph Flow

```
START → entry → should_continue → input_validator → planner → assign_workers → [subtask → combiner] → END
```

**Current Implementation**: The graph implements a fan-out/fan-in pattern:
- **entry**: Checks if thread is already closed (`ended_once` flag). If closed, returns message to use new thread.
- **should_continue**: Conditional edge - returns END if closed, otherwise routes to "input_validator"
- **input_validator**: Scans input for malicious content using pattern matching and optional LLM moderation API
- **planner**: Uses `get_cheapest_model()` to select optimal LLM, generates structured TODOs
- **assign_workers**: Fan-out node that creates parallel tasks for each TODO via `Send()`
- **subtask**: Worker node that processes individual TODOs with model selection
- **combiner**: Fan-in node that synthesizes all completed TODOs into final report

**Execution Details**:
- Uses `Command` objects to return state updates and routing targets
- `Send()` objects enable parallel processing of subtasks
- Circuit breaker pattern with retry logic for LLM failures
- Token usage tracking and execution timing

### Core Components

**1. Graph Definition (`src/task_agent/graph.py`)**
- Defines LangGraph state machine with entry, input_validator, planner, subtask, combiner nodes
- Uses `TaskState` for state management and `Context` for runtime configuration
- Entry point referenced in `langgraph.json` as `./src/task_agent/graph.py:graph`
- Disables LangSmith tracing via environment variable

**2. Node Functions (`src/task_agent/utils/nodes.py`)**
- `entry_node()`: Checks if thread already ended via `ended_once` flag; initializes empty `todos` if not present
- `should_continue()`: Conditional edge function that routes to END or input_validator
- `input_validator_node()`: Validates user input for malicious content using `scan_for_vulnerability()`
- `call_planner_model()`: **Async function** that selects cheapest model, generates structured TODOs, marks thread as ended
- `assign_workers()`: Creates `Send()` objects for each TODO to enable parallel processing
- `call_subtask_model()`: Processes individual TODOs with model selection and retry logic
- `call_combiner_model()`: Synthesizes completed TODOs into final report
- **Important**: Uses a two-step pattern for LLM structured output:
  1. Simple Pydantic schema (`SimpleTODOList`) for LLM output (just `title` and `description`)
  2. Transformation function (`convert_to_todos()`) to fill in full `TODOs` structure with `todo_id`, `todo_completed`, `output` fields
  - This approach avoids LLM issues with complex nested schemas and field validation
- All LLM calls use async/await pattern (`ainvoke` instead of `invoke`)

**3. Input Validation (`src/task_agent/utils/input_validation.py`)**
All validation functions are **async**:

- `scan_for_vulnerability(user_message: str) -> bool`: Main validation function
  - Performs pattern-based detection (shell injection, SQL injection, path traversal, Docker abuse, etc.)
  - Checks for malicious keywords in suspicious context
  - Calls `get_LLM_feedback_on_input()` if `settings.MODERATION_API_CHECK_REQ` is True
  - Returns `True` if safe, `False` if malicious content detected

- `get_LLM_feedback_on_input(prompt: str) -> bool`: Async function using OpenAI's moderation API
  - Uses `AsyncOpenAI` client with `omni-moderation-latest` model
  - Returns `False` if flagged (unsafe), `True` if safe
  - Includes exception handling that fails closed (returns `False` on API errors)

- `get_vulnerability_details(user_message: str) -> dict`: Async function for detailed vulnerability analysis
  - Returns dict with `is_safe`, `detected_issues` list, and `risk_level` ("none", "low", "medium", "high")

**Testing Input Validation**:
```python
# Mock settings.MODERATION_API_CHECK_REQ and LLM feedback in tests
with patch("task_agent.utils.input_validation.settings") as mock_settings:
    mock_settings.MODERATION_API_CHECK_REQ = True
    with patch("task_agent.utils.input_validation.get_LLM_feedback_on_input", return_value=True):
        result = await scan_for_vulnerability("Hello")
```

**4. Circuit Breaker (`src/task_agent/utils/circuit_breaker.py`)**
- `call_llm_with_retry()`: Async function with tenacity retry logic
  - 3 retry attempts with exponential backoff (2-10 seconds)
  - Token usage extraction from response metadata
  - Execution timing and logging
  - Reraises exceptions after retries exhausted
  - Supports fallback model if primary model fails
  - Tracks model usage via `ModelLiveUsage` singleton for cost penalty calculation
- Handles both `usage_metadata` (LangChain 0.1+) and `response_metadata` (older style)

**5. State Management (`src/task_agent/utils/state.py`)**
- `Context`: TypedDict for configurable runtime parameters
- `TaskState`: Main state with `thread_id`, `messages`, `task`, `todos`, `retry_count`, `ended_once`, `completed_todos`, `final_report`
- `todos` field uses `TODOs` Pydantic model from `task_details.py`
- Uses `operator.add` for appending to lists via `Annotated` types
- `ended_once` flag prevents thread reuse after completion

**6. LLM Factory (`src/task_agent/llms/llm_model_factory/llm_factory.py`)**
Uses a registry + resolver pattern for model creation:

**Registry Pattern**:
- Maps provider names (`openai`, `google`, `groq`, `ollama`, `Zhipu`) to LangChain constructor classes
- `LLM_REGISTRY` dictionary contains provider → constructor mappings

**Resolver Logic** (order matters):
1. **Suffix check first**: Models containing `cloud` → Ollama (for local deployment)
2. **Prefix check**: `gpt-` → OpenAI, `gemini-` → Google, `qwen/` or `qwen-` → Groq, `GLM-4.5`/`GLM-4.6V`/`GLM-4.7-Flash` → Zhipu
3. **Default fallback**: `glm-`, `llama`, `gemma` → Ollama

**Factory function**: `create_llm(model: str, **kwargs) -> BaseChatModel`

**Supported Models**:
- OpenAI: gpt-4o, gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-nano
- Google: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-flash-preview:cloud
- Groq: qwen/qwen-2.5-72b-instruct, qwen/qwen3-32b
- Ollama (cloud): qwen3-coder:480b-cloud, gemma3:27b-cloud, glm-4.6:cloud, kimi-k2.5:cloud, gpt-oss:20b-cloud
- Zhipu (z.ai): GLM-4.5-Flash, GLM-4.6V-Flash, GLM-4.7-Flash

**Important**: Models with `cloud` suffix require Ollama running locally:
```bash
# Install Ollama from https://ollama.com
# Pull models
ollama pull qwen3-coder:480b
ollama pull glm-4.6
ollama pull gemma3:27b
ollama pull kimi-k2.5
ollama pull gpt-oss:20b
```

**11. Tools (`src/task_agent/utils/tools.py`)**
- `get_web_search_tool()`: Returns TavilySearch instance for web search capabilities
- Used for LLM tool binding when models need web search functionality
- Configured with max_results=5 and topic="general"

**7. Simplified LLM Selector (`src/task_agent/llms/simple_llm_selector/`)**
Uses LLM-based capability inference for model selection. **All LLM calls are async.**

*Architecture:*
- **Capability Inference** (`inference.py`): Uses Groq model (`qwen/qwen3-32b`) with `async def infer_capabilities()` to analyze task and infer capabilities
- **Capability Matching** (`models.py`): **Loads from CSV files** (`model_capabilities.csv` and `model_costs.csv` in project root)
- **Routing** (`router.py`): `async def select_models()` and `async def get_cheapest_model()` that sort by cost or use coding priority order

*CSV-Based Model Configuration:*
Model capabilities and costs are loaded from CSV files in the project root:
- `model_capabilities.csv`: Columns = `model,reasoning,tools,fast,cheap,informational,coding,vision,long,synthesizing,summarizing,planning`
- `model_costs.csv`: Columns = `model,cost`

**Important**: When adding new models or updating capabilities/costs, edit the CSV files directly - no code changes needed. The loader functions (`_load_model_capabilities_from_csv()` and `_load_model_costs_from_csv()`) automatically find these files at startup.

*Capabilities Inferred:*
- **coding**: Code writing, programming, development
- **informational**: General information, factual queries
- **reasoning**: Complex reasoning, chain-of-thought
- **tools**: Function calling, tool use, API interactions
- **fast**: Low latency, quick response time
- **cheap**: Low cost per token
- **vision**: Image understanding
- **long**: Long context window
- **synthesizing**: Combining information
- **summarizing**: Summarization
- **planning**: Task planning

*Coding Model Priority (in order):*
1. `kimi-k2.5:cloud`
2. `qwen3-coder:480b-cloud`
3. `glm-4.6:cloud`
4. `gemini-2.5-pro`
5. `gemini-2.5-flash`

*Usage (must be async):*
```python
import asyncio
from task_agent.llms.simple_llm_selector import select_models, get_cheapest_model
from task_agent.llms.llm_model_factory.llm_factory import create_llm

async def main():
    # Get top 5 models
    models = await select_models("Write a Python function", top_n=5)

    # Get just the cheapest
    model = await get_cheapest_model("Who invented calculus?")

    # Create and execute (also async)
    llm = create_llm(model, temperature=0.0)
    response = await llm.ainvoke("Who invented calculus?")

asyncio.run(main())
```

**Important**: All selector functions are async. When calling from non-async code, use `asyncio.run()` or `await` within an async context.

**8. Business Objects (`src/task_agent/data_objs/`)**
- **task_details.py**: Defines TODO structures for task planning
  - `TODOs_Output`: Holds execution results (`output`, `model_used`, `execution_time`) - all fields have default empty strings
  - `TODO_details`: Individual TODO with `todo_id`, `todo_name`, `todo_description`, `todo_completed`, `output`
  - `TODOs`: Container with `todo_list` array of `TODO_details` and `thread_id`

**9. External Prompt System (`src/task_agent/llms/prompts/`)**
All system prompts are stored as external `.prompt` files for easy editing, versioning, and collaboration.

*Architecture:*
- **Prompt Files**: Located in `src/task_agent/llms/prompts/` with `.prompt` extension
- **Loader Module**: `__init__.py` provides functions for loading and formatting prompts
- **Template Variables**: Uses `{{variable}}` syntax for parameterization

*Available Prompts:*
- **`planner.prompt`**: Task planning system prompt (generates structured TODOs)
- **`subtask.prompt`**: Worker node prompt for executing individual TODOs
- **`combiner.prompt`**: Synthesizer prompt with `{{user_query}}` template variable
- **`capability_inference.prompt`**: LLM-based capability classifier with `{{task}}` template variable

*Usage:*
```python
from task_agent.llms.prompts import (
    get_planner_prompt,
    get_subtask_prompt,
    get_combiner_prompt,
    get_capability_inference_prompt,
    list_available_prompts
)

# List all available prompts
prompts = list_available_prompts()
# ['capability_inference', 'combiner', 'planner', 'subtask']

# Load without formatting
planner_prompt = get_planner_prompt()

# Load with template variables
combiner_prompt = get_combiner_prompt(user_query="Why is gold price surging?")
capability_prompt = get_capability_inference_prompt(task="Write Python code")
```

*Low-Level API:*
```python
from task_agent.llms.prompts import load_prompt_template, format_prompt

# Load raw template
template = load_prompt_template("combiner")

# Format with variables
formatted = format_prompt(template, user_query="test query")
```

*Benefits:*
- Non-technical team members can edit prompts without touching code
- Git version control for all prompt changes
- Easy A/B testing by swapping prompt files
- Template parameterization for dynamic content

*Adding New Prompts:*
1. Create a new `.prompt` file in `src/task_agent/llms/prompts/`
2. Use `{{variable}}` syntax for template parameters
3. Import and use `get_prompt("your_prompt_name")` in code
4. No code changes needed in the loader module

*Example Prompt File Structure:*
```
You are a {{role}} assistant that helps users with {{domain}}.

Context: {{context}}

Task: {{task}}
```

*Testing:*
```python
# Test file: tests/unit_tests/test_prompt_loading.py
# 39 test cases covering:
# - Loading prompts from files
# - Template variable formatting
# - Error handling for missing prompts
# - Convenience functions
# - Prompt content validation
```

**9. Model Live Usage (`src/task_agent/utils/model_live_usage.py`)**
Tracks model usage counts for cost spreading via exponential penalty:
- `ModelLiveUsage`: Core class tracking usage counts per model via `defaultdict`
- `ModelLiveUsageSingleton`: Thread-safe singleton ensuring single usage tracker instance
- `get_model_usage_singleton()`: Convenience function returning singleton instance

**Usage Tracking**:
- `add_model_usage(model_name, usage)`: Sets or updates usage count (note: sets, doesn't increment)
- `get_model_usage(model_names)`: Returns usage counts (note: parameter name is plural but accepts single string in implementation)
- Singleton pattern ensures consistent tracking across all graph nodes

**10. Router with Cost Penalty (`src/task_agent/llms/simple_llm_selector/router.py`)**
Implements exponential cost penalty for load balancing:
```python
derived_cost = base_cost * exp(COST_SPREADING_FACTOR * model_usage.get_model_usage(model))
```
- As model usage increases, effective cost grows exponentially
- `COST_SPREADING_FACTOR` (default 0.03 in config.py) controls penalty aggression
- Promotes load balancing across multiple models with similar capabilities

### Configuration

Configuration is loaded from environment variables via `pydantic-settings`:
- File: `src/task_agent/config.py`
- Loads from `.env` file
- Required: `OPENAI_API_KEY`, `GOOGLE_API_KEY`
- Optional: `ANTHROPIC_API_KEY` (if using Anthropic models)
- Optional: `TAVILY_API_KEY` (if using web search tools)
- Optional: `INFERENCE_MODEL` (default: `"kimi-k2.5:cloud"`)
- Optional: `MODERATION_API_CHECK_REQ` (default: `True`) - Controls whether LLM moderation API is called
- Optional: `COST_SPREADING_FACTOR` (default: `0.03`) - Controls exponential penalty for model usage
- Optional: `MODEL_COST_CSV_PATH` (default: `"model_costs.csv"`) - Path to model costs CSV file
- Optional: `MODEL_CAPABILITY_CSV_PATH` (default: `"model_capabilities.csv"`) - Path to model capabilities CSV file
- Optional: `API_KEY` (default: `""`) - API key for protected REST endpoints
- Optional: `REQUIRE_AUTH` (default: `false`) - Whether to require authentication for protected endpoints

**Environment Setup:**
```bash
# Required for OpenAI models
OPENAI_API_KEY=sk-...

# Required for Google models
GOOGLE_API_KEY=AIza...

# Optional for Anthropic models
ANTHROPIC_API_KEY=sk-ant-...

# Optional for web search (Tavily)
TAVILY_API_KEY=tvly-...

# Optional: Override default inference model
INFERENCE_MODEL=gpt-4o-mini
MODERATION_API_CHECK_REQ=true

# Optional: Cost spreading for load balancing
COST_SPREADING_FACTOR=0.03

# Optional: CSV file paths (supports absolute, relative, or filename)
MODEL_COST_CSV_PATH=/app/config/model_costs.csv
MODEL_CAPABILITY_CSV_PATH=./config/model_capabilities.csv

# Optional: API Authentication for protected endpoints
API_KEY=your-secret-api-key-here
REQUIRE_AUTH=true
```

**CSV Path Resolution:**
The `MODEL_COST_CSV_PATH` and `MODEL_CAPABILITY_CSV_PATH` settings support flexible path resolution:

1. **Absolute path**: Used directly (e.g., `/app/config/model_costs.csv` for Docker mounts)
2. **Relative path**: Resolved from current working directory (e.g., `./config/costs.csv`)
3. **Filename only**: Resolved from project root (e.g., `model_costs.csv`)

**Docker Examples:**
```bash
# Default: filename only → project root
docker run -e OPENAI_API_KEY=xxx task-graph-engine

# Relative path: from deployment directory
docker run -e OPENAI_API_KEY=xxx \
  -e MODEL_COST_CSV_PATH=./config/costs.csv \
  task-graph-engine

# Absolute path: mounted volume
docker run -e OPENAI_API_KEY=xxx \
  -e MODEL_COST_CSV_PATH=/app/config/model_costs.csv \
  -v /host/config:/app/config \
  task-graph-engine
```

### LangGraph Configuration

- **File**: `langgraph.json`
- **Entry point**: `./src/task_agent/graph.py:graph`
- **Environment**: Uses `.env` file
- **Graph name**: `"agent"` (referenced in LangGraph CLI)
- **Dependency**: Links to `.` (current project) for import resolution

### Key Design Patterns

1. **Async/Await Everywhere**: All LLM calls (`llm.ainvoke()`), selector functions, and validation functions are async
2. **Registry + Resolver**: LLM factory uses registry for provider mapping and resolver for model name parsing
3. **CSV-Based Configuration**: Model capabilities and costs loaded from CSV files at project root for easy maintenance without code changes
4. **External Prompt System**: All system prompts stored as `.prompt` files with `{{variable}}` template syntax for easy editing and versioning
5. **Capability Inference**: Uses LLM to analyze task requirements rather than rule-based matching
6. **State Annotated Lists**: Uses `Annotated[list[BaseMessage], add_messages]` for automatic message merging
7. **Command Pattern**: Node functions return `Command` with update dict and goto target for explicit routing
8. **Send for Fan-out**: `assign_workers` returns list of `Send()` objects to parallelize subtask execution
9. **Context Injection**: Uses `Runtime[Context]` for accessing configurable parameters at runtime
10. **Logging Over Print**: Uses `logging.info()` instead of `print()` throughout; logging configured globally in `logging_config.py`
11. **Circuit Breaker**: Tenacity-based retry with exponential backoff for resilient LLM calls, with fallback model support
12. **Two-Step Structured Output**: Simple Pydantic schema for LLM + transformation function to full structure
13. **Input Validation Pipeline**: Pattern-based detection + keyword context analysis + optional LLM moderation API
14. **Cost-Based Load Balancing**: Exponential penalty on frequently-used models promotes distribution across similar-capability models
15. **Tool Binding**: LLMs can be bound with tools like web search via `llm.bind_tools([get_web_search_tool()])`

### Common Issues

**Circular Import**: If you see circular import errors between `graph.py` and `nodes.py`, ensure `Context` is imported from `task_agent.utils.state` (not from `graph.py`).

**GLM Model Resolution**: The `glm-` prefix resolves to Ollama by default (for local models like `glm-4.6:cloud`). Specific Zhipu GLM models (`GLM-4.5-Flash`, `GLM-4.6V-Flash`, `GLM-4.7-Flash`) are checked first and resolve to the Zhipu provider. Order matters in the prefix dictionary.

**Module Not Found**: If you see `ModuleNotFoundError: No module named 'task_agent'`, run `pip install -e .` to install the package in editable mode.

**Package Name**: The package is installed as `task-graph-engine` (from `pyproject.toml`) but imports use `task_agent` (the Python package name in `src/`).

**Structured Output with LLMs**: When adding new structured output features:
- Use a simple two-step pattern: simple Pydantic schema for LLM + transformation function
- LLMs struggle with complex nested schemas and optional fields
- Keep the LLM schema minimal (only required fields)
- Use transformation functions to fill in default values, IDs, and computed fields
- Field naming matters: the Pydantic field name (`todo_list`) must match what the LLM outputs, not your internal variable name

**Command goto=END Warning**: You may see `Task planner wrote to unknown channel branch:to:END, ignoring it` in logs. This is a benign warning when using `Command(update=..., goto='END')` and can be ignored. The execution completes successfully despite this warning.

**Async/Await Required**: The LLM selector functions (`select_models`, `get_cheapest_model`), all LLM factory invocations, and all input validation functions are async. Always use `await` when calling them within async functions, or wrap with `asyncio.run()` for standalone scripts.

**Logging Configuration**: The project uses Python's `logging` module (not `print` statements). Logging is configured in `src/task_agent/logging_config.py` with colored output via `ColorFormatter` class and initialized in `graph.py` using `setup_logging()`. The setup forces a clean logging environment by removing existing handlers and silencing noisy libraries (uvicorn, httpx, langgraph, langchain, asyncio, httpcore). Use `logging.info()`, `logging.error()`, `logging.debug()` throughout the codebase.

**Fan-out/Fan-in Pattern**: The graph uses `Send()` objects for parallel processing:
- `assign_workers` returns `[Send("subtask", {**state, "todo": td}) for td in todos]`
- Each "subtask" processes independently
- Results accumulate in `completed_todos` list via `operator.add`
- "combiner" node synthesizes all results when all subtasks complete

**Token Usage Tracking**: Token usage is extracted from LLM responses via `_extract_token_usage()` which handles both LangChain 0.1+ `usage_metadata` and older `response_metadata` formats. Not all providers return token usage data.

**Input Validation Mocking in Tests**: When testing functions that call `scan_for_vulnerability()`, you must mock both `settings.MODERATION_API_CHECK_REQ` and `get_LLM_feedback_on_input()` to avoid real API calls. See the test file for examples.

**Cost Spreading Formula**: The router applies an exponential penalty to model costs based on usage: `derived_cost = base_cost × exp(COST_SPREADING_FACTOR × usage_count)`. This promotes load balancing - as a model is used more, its effective cost increases, making other models relatively more attractive. The `COST_SPREADING_FACTOR` (default 0.03, not 0.01 as documented elsewhere) controls how aggressively the penalty ramps up.

**Web Search Tool Binding**: When adding tool support to LLMs in the circuit breaker, use `llm.bind_tools([get_web_search_tool()])` to enable web search capabilities. This requires `TAVILY_API_KEY` in the environment. The tool is defined in `src/task_agent/utils/tools.py` and uses `langchain_tavily`.

**CSV-Based Model Configuration**: Model capabilities and costs are loaded from CSV files (`model_capabilities.csv` and `model_costs.csv`). The path resolution is flexible:
- Default: Looks for files in project root (detected by finding `pyproject.toml` or `langgraph.json`)
- Absolute path: Use `MODEL_COST_CSV_PATH=/app/config/costs.csv` for Docker mounts
- Relative path: Use `MODEL_COST_CSV_PATH=./config/costs.csv` to resolve from cwd
- Filename only: Use `MODEL_COST_CSV_PATH=costs.csv` to resolve from project root

To add new models or update existing ones:
1. Edit the CSV files directly (no code changes needed)
2. Follow the format: `model,cost` for costs; `model,reasoning,tools,fast,cheap,informational,coding,vision,long,synthesizing,summarizing,planning` for capabilities
3. The loader functions (`_load_model_capabilities_from_csv()` and `_load_model_costs_from_csv()`) automatically find these files at startup using the path resolution logic in `resolve_csv_path()`

**EXECUTION SUMMARY Logging**: The combiner node logs an "EXECUTION SUMMARY" when it successfully synthesizes all completed subtasks. This is critical for confirming that the user will receive a response.

**What the EXECUTION SUMMARY indicates**:
- User has received the synthesized final report
- All subtasks completed successfully
- The combiner received a valid text response (not tool calls)
- Total execution time for the combiner node

**What to look for in logs**:
```
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:256 | [COMBINER] Response type: <class 'langchain_core.messages.ai.AIMessage'>, content type: <class 'str'>, content length: 2727
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:272 | ============================================================
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:273 | EXECUTION SUMMARY: : why there is current gold price surge , generate brief report
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:274 | Total TODOs: 5
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:275 | Combiner execution time: 22.14s
2026-02-01 12:31:36.024 | INFO | root | nodes.call_combiner_model:276 | ============================================================
```

**If EXECUTION SUMMARY is NOT logged**, the user did NOT receive the response. This happens when:
- The combiner's LLM returned tool calls instead of text content
- `response.content` was empty even though tokens were generated
- The combiner exited early without setting the `final_report`

**How the fix works**:
- The `call_llm_with_retry()` function in `circuit_breaker.py` has a `bind_tools_flag: bool = True` parameter
- By default, ALL LLM calls bind web search tools (for subtasks that may need web search)
- The combiner explicitly passes `bind_tools_flag=False` to skip tool binding
- This ensures the combiner always gets text responses instead of tool calls

**Debug logging**:
If the EXECUTION SUMMARY is missing, check for these log lines:
- `[COMBINER] Response type: ...` - Shows the response object type and content length
- `Empty final_output. Response: ...` - Indicates the response had no text content
- `Response contains tool calls instead of text: ...` - Indicates tool binding issue

**Key implementation detail**: In `src/task_agent/utils/nodes.py`, the combiner calls:
```python
response: AIMessage = await call_llm_with_retry(
    cheapest,
    prompt,
    fallback_model="gpt-4o-mini",
    temperature=0.0,
    bind_tools_flag=False  # Don't bind tools for combiner
)
```

**External Prompt System**: All system prompts are stored in `src/task_agent/llms/prompts/` as `.prompt` files:
- **Edit prompts**: Modify files directly - no code changes needed
- **Template syntax**: Use `{{variable}}` for parameters (e.g., `{{user_query}}`, `{{task}}`)
- **Add new prompts**: Create `your_prompt.prompt` and use `get_prompt("your_prompt")`
- **Convenience functions**: `get_planner_prompt()`, `get_subtask_prompt()`, `get_combiner_prompt(user_query)`, `get_capability_inference_prompt(task)`
- **List available**: `list_available_prompts()` returns all prompt names
- **Error handling**: `FileNotFoundError` raised if prompt file doesn't exist (includes available prompts in error message)
