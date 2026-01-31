# 🧠 Task Graph Engine

> *"Because choosing the right AI model shouldn't require a PhD in economics"* 🎓💸

Meet **Task Graph Engine** - a LangGraph-based task planning system that thinks before it spends. We use intelligent LLM selection to match your tasks with the most appropriate (and cheapest!) model automatically. 💰✨

---

## ✨ What Makes This Special?

### 🎯 The Problem We Solve
You've got 14 LLMs to choose from. Do you use GPT-4o for that simple summary? Gemini Flash for complex coding? Or maybe Ollama's local models for... something? 🤔

**We fix this analysis paralysis** by:
1. **Inferring** what your task actually needs (coding? reasoning? just speed?)
2. **Selecting** the best model for the job from 14 providers
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

#### 📋 **CSV-Based Model Configuration** 🆕
Model capabilities and costs are loaded from CSV files - no code changes needed:
```bash
# model_capabilities.csv
model,reasoning,tools,fast,cheap,informational,coding,vision,long
gpt-4o,True,True,True,False,True,True,True,True,True

# model_costs.csv
model,cost
gpt-4o,5.0
gpt-4o-mini,0.035
```

**Flexible Path Resolution**:
- Default: Project root (`model_costs.csv`)
- Relative: Current directory (`./config/costs.csv`)
- Absolute: Docker mount (`/app/config/costs.csv`)

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
19 models across 5 providers, one clean interface:
| Provider | Models | Specialty |
|----------|--------|-----------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-5-* | General purpose |
| **Google** | gemini-2.5-*/flash-lite/pro, gemini-3-* | Speed & reasoning |
| **Groq** | qwen-2.5, qwen3 | Fast inference |
| **Ollama** | llama, gemma3, glm-4.6:cloud, qwen3-coder, kimi-k2.5 | Local deployment |
| **Zhipu** | GLM-4.5/4.6V/4.7-Flash | Chinese models |

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- API keys for at least one provider (OpenAI, Google, Groq)
- [Ollama](https://ollama.com) (optional, for local models)

### Quick Start
```bash
# Clone and install
git clone https://github.com/yourusername/task-graph-engine.git
cd task-graph-engine
pip install -e .

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Start LangGraph dev server
langgraph dev

# For non-standard LangChain models (community models, z.ai, nvidia, etc.)
# that use synchronous calls, use --allow-blocking flag:
langgraph dev --allow-blocking
```

> **Note**: Some LangChain community integrations (like `langchain-nvidia-ai-core`, `langchain-community`, `z.ai`) use synchronous HTTP calls internally. The `--allow-blocking` flag prevents LangGraph from throwing warnings about blocking calls in an async context.
>
> **Standard models** (OpenAI, Google/Gemini, Groq, Anthropic) work fine without this flag since they support async/await.

Visit http://127.0.0.1:2024 and start planning tasks! 🎉

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
┌─────────┐     ┌──────────────┐     ┌──────────────────┐
│  START  │────▶│    entry     │────▶│ should_continue  │─────────────────┐
└─────────┘     └──────────────┘     └──────────────────┘                 │
                     │                       │                            │
                     │                       ▼                            ▼
                     │                ┌─────────────────┐           ┌──────────────┐
                     │                │ input_validator │──────────▶│   planner    │──┐
                     │                └─────────────────┘           └──────────────┘  │
                     │                                                              │
                     │                                                              ▼
                     └─────────────────────────────────────────────────────────── END  │
                                                                                      │
                                                                                      ▼
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  assign_workers │────▶│   subtask   │────▶│  combiner    │────▶│        END           │
└─────────────────┘     └─────────────┘     └──────────────┘     └──────────────────────┘
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
1. START → entry node
2. entry → conditional check with should_continue
3. If should_continue returns "input_validator", proceed to input validation
4. input_validator → planner (generates TODOs)
5. planner → assign_workers (fan-out to multiple subtasks)
6. Each subtask → combiner (fan-in after all subtasks complete)
7. combiner → END

### Key Components
- **`graph.py`**: LangGraph state machine definition
- **`nodes.py`**: Node functions (entry, planner, subtask, combiner, input validation)
- **`llm_factory.py`**: Multi-provider model creation
- **`simple_llm_selector/`**: Capability inference + routing
- **`state.py`**: TypedDict state management
- **`task_details.py`**: Pydantic models for TODOs
- **`circuit_breaker.py`**: Retry logic with fallback models
- **`input_validation.py`**: Scanning for potentially malicious content

---

## 🧪 Development

### Code Quality
```bash
# Linting (ruff - fast and opinionated)
ruff check .              # Find problems
ruff check --fix .        # Fix them automatically

# Type checking (mypy)
mypy src/                 # Catch type errors before runtime
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

**Test Stats**: 🧪 299 test cases, covering edge cases like special characters, long inputs, model resolution logic, and CSV path resolution.

---

## 📊 Configuration

All settings via environment variables (`.env` file):

```bash
# Required 🔐
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional (inference model)
INFERENCE_MODEL=kimi-k2.5:cloud

# Optional (moderation)
MODERATION_API_CHECK_REQ=true

# Optional (cost spreading for load balancing)
COST_SPREADING_FACTOR=0.03

# Optional (CSV file paths - supports absolute, relative, or filename)
MODEL_COST_CSV_PATH=model_costs.csv
MODEL_CAPABILITY_CSV_PATH=model_capabilities.csv
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

1. **Async Everywhere** - Why block when you can await? ⏳
2. **Type Safety** - TypedDict + Pydantic = fewer runtime surprises 🎯
3. **Logging Over Print** - Structured logs with thread IDs 📝
4. **Simplicity Wins** - Two-step transformation over complex schemas 🎭
5. **Capability Inference** - Let AI figure out what AI you need 🤖

---

## 🚧 Current Limitations

- **No Exponential Backoff**: Basic `retry_count` tracking but no exponential backoff yet
- **Local Models Only**: Ollama requires running models locally (no remote API)
- **E2E Tests Skipped**: End-to-end tests need API keys to run (marked as skip by default)

---

## 🗺️ Roadmap

- [ ] **Multi-Agent Evaluation**: Implement `CombinedPlan` for parallel agent evaluation
- [ ] **Circuit Breakers**: Add resilience patterns for LLM API failures
- [ ] **Cost Tracking**: Log token usage and costs per task 💰
- [ ] **Docker Support**: Containerize for easy deployment 🐳
- [ ] **Monitoring**: OpenTelemetry metrics and tracing 📊
- [ ] **Rate Limiting**: Per-user quotas to prevent bill shock 🛡️

---

## 📜 License

MIT License - feel free to use this in your own projects!

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
