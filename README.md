# 🧠 Task Graph Engine

> *"Because choosing the right AI model shouldn't require a PhD in economics"* 🎓💸

Meet **Task Graph Engine** - a LangGraph-based task planning system that thinks before it spends. We use intelligent LLM selection to match your tasks with the most appropriate (and cheapest!) model automatically. 💰✨

---

## ✨ What Makes This Special?

### 🎯 The Problem We Solve
You've got 25 LLMs to choose from. Do you use GPT-4o for that simple summary? Gemini Flash for complex coding? Or maybe Ollama's local models for... something? 🤔

**We fix this analysis paralysis** by:
1. **Inferring** what your task actually needs (coding? reasoning? just speed?)
2. **Selecting** the best model for the job from 25 models across 6 providers
3. **Optimizing** for cost without sacrificing quality

### 🚀 Novel Architecture Highlights

#### 🧠 **LLM-Based Capability Inference** 🆕
Instead of brittle rule-based matching, we use an LLM to analyze your task and infer required capabilities:
```python
# Not this ❌
if "code" in task.lower():
    model = "gpt-4o"

# But this ✅
capabilities = await infer_capabilities("Write a Python function")
# → {coding: 0.95, reasoning: 0.8, fast: 0.3}
model = await get_cheapest_model(capabilities)
# → qwen3-coder:480b-cloud (cheapest coding model!)
```

#### 🏭 **Registry + Resolver Factory Pattern**
A clever factory that handles model name resolution through layered logic:
```python
create_llm("gemini-2.5-flash")        # → Google (prefix match)
create_llm("qwen/qwen-2.5-72b")       # → Groq (prefix match)
create_llm("gemma3:27b-cloud")        # → Ollama (suffix "cloud")
create_llm("GLM-4.5-Flash")           # → Zhipu (prefix match)
create_llm("llama-3.2")               # → Ollama (default fallback)
```
**Order matters!** We check suffix → prefix → default to handle edge cases elegantly.

**Ollama Cloud URL Support** 🆕:
For Docker deployments or machines without Ollama Desktop, enable cloud URL mode:
```bash
# In .env or docker-compose.yml
USE_OLLAMA_CLOUD_URL=true
OLLAMA_CLOUD_URL=https://your-ollama-cloud.com
OLLAMA_API_KEY=your-api-key-here
```

When enabled, all Ollama models (those with `:cloud` suffix or resolved to Ollama) will use the remote endpoint with proper authentication instead of the local `http://localhost:11434`.

#### 📋 **CSV-Based Model Configuration** 🆕
Model capabilities and costs are loaded from CSV files - no code changes needed:
```bash
# model_capabilities.csv (includes 'enabled' column for model availability)
model,reasoning,tools,fast,cheap,informational,coding,vision,long,synthesizing,summarizing,planning,enabled
gpt-4o,True,True,True,False,True,True,True,True,True,True,False,True
GLM-4.7-Flash,True,True,True,True,True,False,False,False,False,False,False,False

# model_costs.csv
model,cost
gpt-4o,5.0
gpt-4o-mini,0.035
```

**Model Availability Control**:
- Set `enabled=False` to disable specific models
- Disabled models are excluded from selection automatically
- The `FALLBACK_MODEL` (default: `gpt-4o-mini`) overrides disabled status
- Useful for testing, cost control, or provider issues

**Flexible Path Resolution**:
- Default: Project root (`model_costs.csv`)
- Relative: Current directory (`./config/costs.csv`)
- Absolute: Docker mount (`/app/config/costs.csv`)

#### 📊 **Dynamic Cost Calculation with Weighted Penalties** 🆕
The router tracks token usage and call frequency per model, applying configurable weighted penalties:

```python
# Token usage tracking
model_usage.add_model_token_usage(model_name, token_count)

# Dynamic cost calculation with weighted penalties
derived_cost = base_cost × (w₁ × ln(10 + α × call_count) + w₂ × log_b(token_usage))
```

**Where**:
- `w₁` = `FORMULA_WEIGHT_CALL_COUNT` (default: 0.5) - weight for call frequency
- `w₂` = `FORMULA_WEIGHT_TOKEN_COUNT` (default: 0.5) - weight for token volume
- `α` = `COST_SPREADING_FACTOR` (default: 0.03) - spreading aggression
- `b` = `TOKEN_USAGE_LOG_BASE` (default: 100.0) - log scaling base

**How it works**:
- Tracks both call frequency and token consumption per model
- Logarithmic scaling ensures gentle, predictable penalty growth
- Additive combination means same penalty for all models regardless of base cost
- Configurable weights allow tuning for rate limiting vs cost optimization

**Configuration**:
```bash
# Penalty weights (must sum to 1.0 for consistent behavior)
FORMULA_WEIGHT_CALL_COUNT=0.5    # Weight for call frequency penalty
FORMULA_WEIGHT_TOKEN_COUNT=0.5   # Weight for token volume penalty

# Scaling parameters
TOKEN_USAGE_LOG_BASE=100.0       # Base for log scaling
COST_SPREADING_FACTOR=0.03       # Penalty aggression
```

**Tuning strategies**:
- **Call-focused** (w₁=0.7, w₂=0.3): Prevent API throttling, prioritize request distribution
- **Token-focused** (w₁=0.3, w₂=0.7): Optimize for actual dollar cost
- **Balanced** (w₁=0.5, w₂=0.5): Equal emphasis on both factors (default)

#### 📝 **External Prompt System** 🆕
All system prompts are stored as external `.prompt` files - no code changes needed:
```bash
src/task_agent/llms/prompts/
├── planner.prompt              # Task planning prompt
├── subtask.prompt              # Worker node prompt
├── combiner.prompt             # Synthesizer with {{user_query}} template
└── capability_inference.prompt # Capability classifier with {{task}} template
```

**Usage**:
```python
from task_agent.llms.prompts import (
    get_planner_prompt,
    get_combiner_prompt,
    list_available_prompts
)

# List all prompts
print(list_available_prompts())
# ['capability_inference', 'combiner', 'planner', 'subtask']

# Use prompts
planner = get_planner_prompt()
combiner = get_combiner_prompt(user_query="Analyze market trends")
```

**Benefits**:
- Non-technical team can edit prompts directly
- Git version control for all changes
- Easy A/B testing by swapping files
- Template variables with `{{variable}}` syntax

#### 👁️ **Vision-Capable Models** 🆕
Several models support vision capabilities for image analysis (when implemented):
- **gpt-4o-mini**: OpenAI's cost-effective vision model
- **gemini-2.5-flash**: Google's fast vision model
- **gemini-2.5-flash-lite**: Google's cheapest vision option
- **gemini-2.5-pro**: Google's high-end vision model
- **gemini-3-flash-preview:cloud**: Latest Gemini vision via Ollama
- **gemma3:27b-cloud**: Gemma vision model via Ollama
- **gemma-3-27b-it**: Gemma vision via Google
- **gemma-3-12b-it**: Smaller Gemma vision via Google

**Note**: Current implementation is **text-only**. Vision capability exists in model metadata but is not actively used for image processing.

#### 🔄 **Two-Step Structured Output**
LLMs hate complex nested schemas. So we trick them:
```python
# Step 1: LLM gets simple schema
SimpleTODOList(title, description)

# Step 2: We transform to full structure
TODOs(todo_id, title, description, completed, output, metadata)
```
**Result**: Better reliability, less fighting with GPT-4's validation demons. 👹

#### ⚡ **Async-First Architecture**
Every LLM call is async. Every selector function is async. We don't block:
```python
# All async, all the time 🏃‍♂️💨
models = await select_models("Write Python code")
model = await get_cheapest_model("What's 2+2?")
response = await llm.ainvoke(messages)
```

#### 🌐 **Multi-Provider Support**
25 models across 6 providers, one clean interface:
| Provider | Models | Specialty |
|----------|--------|-----------|
| **OpenAI** | gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-nano | General purpose |
| **Google** | gemini-2.5-*/flash-lite/pro, gemini-3-*, gemma-3-* | Speed & reasoning |
| **Groq** | qwen-2.5, qwen3, llama-3.* | Fast inference |
| **Ollama** | gemma3, glm-4.6:cloud, qwen3-coder, kimi-k2.5, deepseek-v3.1, gpt-oss | Local deployment |
| **Zhipu** | GLM-4.5/4.6V/4.7-Flash | Chinese models |
| **Sarvam AI** | sarvam-m | Default inference model |

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+ (if you're on 3.10, we won't tell anyone—but upgrade anyway)
- API keys for at least one provider (OpenAI, Google, Groq)—the bouncers won't let you in without one
- [Ollama](https://ollama.com) (optional, for running models locally and feeling like a hacker)

### Quick Start
```bash
# Clone and install
git clone https://github.com/yourusername/task-graph-engine.git
cd task-graph-engine
pip install -e .  # The "-e" stands for "editable", not "evil"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys (guard them with your life)

# Start LangGraph dev server
# Use --allow-blocking for community models (z.ai, nvidia, etc.)
langgraph dev --allow-blocking  # Watch the magic happen

```

> **Note**: Some LangChain community integrations (like `langchain-nvidia-ai-core`, `langchain-community`, `z.ai`) use synchronous HTTP calls internally. The `--allow-blocking` flag prevents LangGraph from throwing warnings about blocking calls in an async context.
>
> **Standard models** (OpenAI, Google/Gemini, Groq, Anthropic) work fine without this flag since they support async/await.

**Server URLs**:
- API: http://127.0.0.1:2024
- Interactive Docs: http://127.0.0.1:2024/docs
- LangSmith Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024

## Dockerization:
1. fill .env with required AUTH keys or override properties from config.py
2. start docker backend or confirm docker is running in the background
3. langgraph up --recreate
4. docker ps
5. peek inside container :
   docker logs -f task-graph-engine-langgraph-api-1 (or something similar)
6. update connecting url from langsmith to connect or check
http://localhost:8123/docs
7. in case docker deployment failing for dependencies, add in .env file
PYTHONPATH=/deps/task-graph-engine/src

### Docker Environment Variables

All environment variables from `.env.example` can be passed to Docker:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key |
| `GOOGLE_API_KEY` | Yes | - | Google API key |
| `ANTHROPIC_API_KEY` | No | - | Anthropic API key |
| `TAVILY_API_KEY` | No | - | Tavily web search API key |
| `INFERENCE_MODEL` | No | `sarvam-m` | Default inference model |
| `MODERATION_API_CHECK_REQ` | No | `true` | Enable LLM moderation API |
| `COST_SPREADING_FACTOR` | No | `0.03` | Load balancing penalty factor |
| `USE_OLLAMA_CLOUD_URL` | No | `false` | Enable Ollama cloud URL for remote deployments |
| `OLLAMA_CLOUD_URL` | No | `https://ollama.com` | Ollama cloud endpoint URL |
| `OLLAMA_API_KEY` | No | - | API key for Ollama cloud endpoint |
| `API_KEY` | No | - | API authentication key |
| `REQUIRE_AUTH` | No | `false` | Enable endpoint authentication |

#### Health Check

The Docker container includes a built-in health check:

```bash
# Check container health
docker ps

# Manual health check
curl http://localhost:2024/api/health
```

---

## 🌐 REST API

The server includes a REST API with both LangGraph core endpoints and custom endpoints for monitoring and configuration.

### Base URL
```
http://127.0.0.1:2024
```

### Available Endpoints

#### Custom Endpoints (`/api/*`)

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | API information | No |
| GET | `/api/health` | Health check and system info | No |
| GET | `/api/models` | List all models with capabilities & costs | Optional |
| GET | `/api/models/{model_name}` | Get details for specific model | Optional |
| GET | `/api/statistics` | Runtime statistics (model usage, costs) | Optional |
| GET | `/api/config` | Current configuration settings | Optional |

#### LangGraph Core Endpoints

| Endpoint | Description |
|----------|-------------|
| `/ok` | Health check |
| `/runs` | Create and manage runs |
| `/threads` | Thread management |
| `/assistants` | Assistant configuration |
| `/docs` | Interactive API documentation (Scalar) |

### Authentication

Protected endpoints require API key authentication when enabled:

```bash
# Set in .env file
API_KEY=your-secret-api-key-here
REQUIRE_AUTH=true
```

**Usage**:
```bash
# Without auth (REQUIRE_AUTH=false or not set)
curl http://127.0.0.1:2024/api/models

# With auth (REQUIRE_AUTH=true)
curl -H "Authorization: Bearer your-secret-api-key-here" \
  http://127.0.0.1:2024/api/models
```

### Example Responses

**Health Check**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "auth_required": false,
  "models_loaded": 25
}
```

**List Models**:
```json
{
  "count": 25,
  "models": {
    "gpt-4o-mini": {
      "capabilities": ["cheap", "coding", "fast", "informational", "reasoning", "tools", "vision", "long", "synthesizing", "summarizing"],
      "cost": 1.35,
      "is_coding_priority": false
    },
    "qwen3-coder:480b-cloud": {
      "capabilities": ["cheap", "coding", "fast", "informational", "reasoning"],
      "cost": 0.013,
      "is_coding_priority": true
    }
  }
}
```

**Statistics** (after some usage):
```json
{
  "cost_spreading_factor": 0.03,
  "total_models_with_usage": 2,
  "models": {
    "llama-3.3-70b-versatile": {
      "usage_count": 3,
      "base_cost": 0.012,
      "derived_cost": 0.0131,
      "penalty_factor": 1.0917
    }
  }
}
```

---

## 🎬 Quick Demo

### See the Magic Happen
```bash
# Watch model selection in action
python src/task_agent/llms/simple_llm_selector/demo.py

# See the full integration
python src/task_agent/llms/demo_integration.py
```

### Example Usage
```python
import asyncio
from task_agent.llms.simple_llm_selector import get_cheapest_model
from task_agent.llms.llm_model_factory.llm_factory import create_llm

async def main():
    # Find the best model for coding
    model = await get_cheapest_model("Write a REST API in Python")
    print(f"Selected: {model}")  # → qwen3-coder:480b-cloud

    # Create and execute
    llm = create_llm(model, temperature=0.0)
    response = await llm.ainvoke("Write a Flask API with /hello endpoint")
    print(response.content)

asyncio.run(main())
```

**Output**:
```
✨ Inferring capabilities for: "Write a REST API in Python"
🎯 Detected: coding=0.92, reasoning=0.75, fast=0.40
💰 Selected cheapest coding model: qwen3-coder:480b-cloud ($0.12/1M tokens)
```

---

## 🏗️ Architecture

### Graph Flow
```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  START  │────▶│    entry     │────▶│ should_continue  │────▶│   input_validator    │
└─────────┘     └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                                                │
                                                                                ▼
                                                                      ┌─────────────────────┐
                                                                      │route_after_validation│
                                                                      └─────────────────────┘
                                                                                │
                                                                   ┌────────────┴────────┐
                                                                   ▼                     ▼
                                                              ┌─────────┐          ┌─────┐
                                                              │ planner │          │ END │
                                                              └────┬────┘          └─────┘
                                                                   │
                                                                   ▼
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌─────┐
│  assign_workers │────▶│   subtask   │────▶│  combiner    │────▶│     end      │────▶│ END │
└─────────────────┘     └─────────────┘     └──────────────┘     └──────────────┘     └─────┘
         │                     │                                       │
         └─────────────────────┼───────────────────────────────────────┘
                               │
                         ┌─────▼─────┐
                         │ fan-out   │
                         │ (for each │
                         │  TODO)    │
                         └───────────┘
```

The actual flow is:
1. START → entry node (initializes state, sets start_time)
2. entry → conditional check with should_continue
3. If thread not closed → input_validator (security check)
4. input_validator → route_after_validation (checks input_valid flag)
5. If valid → planner (generates TODOs), else → END
6. planner → assign_workers (fan-out to multiple subtasks)
7. Each subtask → combiner (fan-in after all subtasks complete)
8. combiner → end (logs execution summary with total time)
9. end → END

### Key Components
- **`graph.py`**: LangGraph state machine definition
- **`nodes.py`**: Node functions (entry, planner, subtask, combiner, end, input validation, routing)
- **`llm_factory.py`**: Multi-provider model creation
- **`simple_llm_selector/`**: Capability inference + routing
- **`prompts/`**: External prompt files with template variables
- **`state.py`**: TypedDict state management
- **`task_details.py`**: Pydantic models for TODOs
- **`circuit_breaker.py`**: Retry logic with fallback models
- **`input_validation.py`**: Scanning for potentially malicious content
- **`src/webapp.py`**: Custom FastAPI app with REST API endpoints (`/api/*`)

---

## 🖥️ Streamlit Web UI

A user-friendly web interface for testing the Task Graph Engine with text-only queries.

### Features

- **Text Queries**: Submit tasks and questions in natural language
- **Real-time Progress**: Watch task execution with live status updates
- **Model Info**: View available models and their capabilities
- **Thread Management**: Create new threads or continue conversations
- **Results Display**: Clean presentation of final reports with execution details

### Running the UI

```bash
# 1. Start the LangGraph server (in one terminal)
langgraph dev

# 2. Start the Streamlit UI (in another terminal)
streamlit run ui/app.py
```

**Access the UI**: http://localhost:8501

### Example Queries

- "Analyze why gold prices are surging and what the trends suggest"
- "Write a Python function to validate email addresses"
- "Compare quantum computing vs classical computing approaches"
- "Explain the principles of RESTful API design"

### Screenshot Preview

The UI displays:
- **Sidebar**: System info, available models, capabilities overview
- **Main Area**: Text input area for queries
- **Results**: Final report with expandable execution details

**Note**: The UI currently supports text-only input. Images are not processed.

---

## 🧪 Development

### Code Quality
```bash
# Linting (ruff - fast and opinionated)
ruff check .              # Find problems (it will, aggressively)
ruff check --fix .        # Fix them automatically (like magic, but real)

# Type checking (mypy)
mypy src/                 # Catch type errors before runtime catches you
```

### Testing
```bash
# Run all tests
pytest

# Unit tests only (fast, no API keys needed)
pytest tests/unit_tests/

# End-to-end tests (requires API keys!)
pytest -m '' tests/end_to_end/

# Specific test file
pytest tests/unit_tests/test_combiner.py -v
```

**Test Stats**: 🧪 434 test cases covering edge cases like special characters, long inputs, model resolution logic, CSV path resolution, token usage tracking, prompt loading/formatting, and capability-based tool binding. That's more test coverage than your ex has commitment issues.

---

## 📊 Configuration

All settings via environment variables (`.env` file):

```bash
# Required 🔐
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional (inference model) - default is sarvam-m
INFERENCE_MODEL=sarvam-m

# Optional (fallback model - overrides enabled flag if disabled)
FALLBACK_MODEL=gpt-4o-mini

# Optional (moderation)
MODERATION_API_CHECK_REQ=true

# Optional (cost spreading for load balancing)
COST_SPREADING_FACTOR=0.03

# Optional (CSV file paths - supports absolute, relative, or filename)
MODEL_COST_CSV_PATH=model_costs.csv
MODEL_CAPABILITY_CSV_PATH=model_capabilities.csv

# Optional (Ollama cloud URL for Docker/remote deployments)
USE_OLLAMA_CLOUD_URL=true
OLLAMA_CLOUD_URL=https://your-ollama-cloud.com
OLLAMA_API_KEY=your-ollama-api-key

# Optional (API authentication for protected endpoints)
API_KEY=your-secret-api-key-here
REQUIRE_AUTH=false
```

**CSV Path Resolution** 🆴:
- **Absolute path**: `/app/config/model_costs.csv` (use for Docker mounts)
- **Relative path**: `./config/model_costs.csv` (resolved from cwd)
- **Filename only**: `model_costs.csv` (resolved from project root)

**Docker Example**:
```bash
docker run -e OPENAI_API_KEY=xxx \
  -e MODEL_COST_CSV_PATH=/app/config/model_costs.csv \
  -v /host/config:/app/config \
  task-graph-engine
```

**No hardcoded secrets** - we use `pydantic-settings` for type-safe config loading. 🔒

---

## 🎨 Design Philosophy

1. **Async Everywhere** - Why block when you can await? We're not savages. ⏳
2. **Type Safety** - TypedDict + Pydantic = fewer runtime surprises (like finding out your "string" was actually a None) 🎯
3. **External Prompts** - Edit system prompts without touching code. Your PM will thank you. 📝
4. **Logging Over Print** - Structured logs with thread IDs. `print()` is for scripts, not serious software. 📝
5. **Simplicity Wins** - Two-step transformation over complex schemas. LLMs have enough trouble with basic instructions; let's not torture them with nested schemas. 🎭
6. **Capability Inference** - Let AI figure out what AI you need. Inception, but productive. 🤖

---

## 🚧 Current Limitations

- **E2E Tests Skipped**: End-to-end tests need API keys to run (marked as skip by default)—we'd rather not accidentally spend your rent money on LLM calls
- **Vision Model Quirks**: Sometimes models see things that aren't there. Like your imagination, but worse.

---

## 🔍 Troubleshooting & Monitoring

### EXECUTION SUMMARY Logging

The most important log line to monitor is the **EXECUTION SUMMARY**. This confirms that the user received a synthesized response and shows the total execution time.

**What to look for in logs**:
```
2026-02-06 12:31:36.024 | INFO | root | nodes.end_node:307 | ============================================================
2026-02-06 12:31:36.024 | INFO | root | nodes.end_node:308 | EXECUTION SUMMARY: Analyze gold price surge
2026-02-06 12:31:36.024 | INFO | root | nodes.end_node:309 | Total TODOs: 5
2026-02-06 12:31:36.024 | INFO | root | nodes.end_node:310 | Total task time: 35.42s
2026-02-06 12:31:36.024 | INFO | root | nodes.end_node:311 | ============================================================
```

**If EXECUTION SUMMARY is NOT logged**, the user did NOT receive the response. This typically happens when:
- The input validation failed (malicious content detected)
- The planner failed to generate TODOs
- The combiner exited early without setting the `final_report`
- The graph was terminated before reaching the `end_node`

**Technical Details**:

The system uses web search tools for subtasks that may need current information. However, not all models support function calling. The implementation now:

1. **Checks model capabilities**: `call_llm_with_retry()` verifies `"tools" in get_model_capabilities(model_name)` before binding
2. **Smart tool binding**: Only models with `tools=True` in the CSV get web search tools bound
3. **Combiner exception**: The combiner passes `bind_tools_flag=False` to skip tool binding entirely
4. **Result**: Models without tool capability (like `qwen3-coder:480b-cloud`, `gemma-3-27b-it`, `glm-4.6:cloud`) don't fail with tool-binding errors

**Models WITH tools capability** (19 models): gpt-4o-mini, gpt-5-mini, gpt-5-nano, gpt-4.1-nano, gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-3-flash-preview:cloud, qwen/qwen-2.5-72b-instruct, qwen/qwen3-32b, qwen3-coder:480b-cloud, gemma3:27b-cloud, GLM-4.5-Flash, GLM-4.6V-Flash, GLM-4.7-Flash, llama-3.3-70b-versatile, llama-3.1-8b-instant, kimi-k2.5:cloud, deepseek-v3.1:671b-cloud

**Models WITHOUT tools capability** (5 models): moonshotai/kimi-k2-instruct, gemma-3-27b-it, gemma-3-12b-it, glm-4.6:cloud, gpt-oss:20b-cloud

**Implementation** (`src/task_agent/utils/nodes.py`):
```python
# Combiner explicitly skips tool binding
response: AIMessage = await call_llm_with_retry(
    cheapest,
    prompt,
    fallback_model="gpt-4o-mini",
    temperature=0.0,
    bind_tools_flag=False  # Don't bind tools for combiner
)
```

**Debug logging** includes:
- Response type and content length
- Warnings for empty content or tool calls
- Error messages if the response format is unexpected

### External Prompt System

All system prompts are stored as `.prompt` files in `src/task_agent/llms/prompts/` for easy editing without code changes.

**Available Prompts**:
- `planner.prompt` - Task planning system prompt
- `subtask.prompt` - Worker node prompt
- `combiner.prompt` - Synthesizer with `{{user_query}}` template variable
- `capability_inference.prompt` - Capability classifier with `{{task}}` template variable

**Editing Prompts**:
```bash
# Edit a prompt file directly
vim src/task_agent/llms/prompts/combiner.prompt

# Changes take effect on next run (no restart needed for langgraph dev)
```

**Adding New Prompts**:
1. Create `your_prompt.prompt` in `src/task_agent/llms/prompts/`
2. Use `{{variable}}` syntax for template parameters
3. Use `get_prompt("your_prompt", variable="value")` in code

**Example Prompt File**:
```
You are a {{role}} assistant for {{domain}}.

Guidelines:
{{guidelines}}

Task: {{task}}
```

---

## 🗺️ Roadmap

- [x] **Circuit Breakers**: Retry logic with exponential backoff for LLM API failures ✅ (because things *will* fail)
- [x] **Cost Tracking**: Token usage and cost logging per task 💰 (so you know exactly how much this brilliance cost)
- [x] **Docker Support**: Containerize for easy deployment 🐳 (works on my machine → works in the container → hopefully works in production)
- [x] **REST API**: Custom endpoints for monitoring and configuration 🌐 (API-first, always)
- [x] **Streamlit UI**: Web interface for testing 🖥️ (for when you're tired of curl commands)
- [x] **Capability-Based Tool Binding**: Only bind tools to models that support function calling 🔧 (prevents errors with non-tool-capable models)
- [ ] **Image Analysis**: Multimodal support for vision-capable models 👁️ (what does this meme *mean*?)
- [ ] **Multi-Agent Evaluation**: Implement `CombinedPlan` for parallel agent evaluation (divide and conquer, but make it AI)
- [ ] **Monitoring**: OpenTelemetry metrics and tracing 📊 (because "it's slow" is not a helpful bug report)
- [ ] **Rate Limiting**: Per-user quotas to prevent bill shock 🛡️ (your wallet will thank us)

---

## 📜 License

MIT License - feel free to use this in your own projects! No credit required, but appreciated (we're watching... just kidding, or are we?)

---

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM abstraction layer
- [Pydantic](https://docs.pydantic.dev/) - Data validation and settings
- [Groq](https://groq.com/) - Blazing fast inference
- [Ollama](https://ollama.com/) - Local model deployment

---

<div align="center">

**Made with ❤️ and too much coffee**

*"The best model is the one you didn't have to choose yourself"* ☕

</div>
