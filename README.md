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
create_llm("llama-3.2")               # → Ollama (default fallback)
```
**Order matters!** We check suffix → prefix → default to handle edge cases elegantly.

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
14 models across 4 providers, one clean interface:
| Provider | Models | Specialty |
|----------|--------|-----------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-5-* | General purpose |
| **Google** | gemini-2.5-flash/pro, gemini-3-* | Speed & reasoning |
| **Groq** | qwen-2.5, qwen3, qwen3-coder | Fast inference |
| **Ollama** | llama, gemma3, glm-4.6:cloud | Local deployment |

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
```

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
┌─────────┐     ┌──────────────────┐     ┌─────────┐
│  START  │────▶│      entry       │────▶│  END    │
└─────────┘     └──────────────────┘     └─────────┘
                     │
                     ▼
              ┌──────────────┐
              │ should_cont. │─────┐ (thread closed)
              └──────────────┘     │
                     │             │
                     ▼             │
              ┌──────────────┐     │
              │   planner    │◀────┘
              └──────────────┘
                   │
                   ▼
         🧠 Select cheapest model
                   │
                   ▼
         📋 Generate TODOs (async)
                   │
                   ▼
              ┌─────────┐
              │   END   │
              └─────────┘
```

### Key Components
- **`graph.py`**: LangGraph state machine definition
- **`logic.py`**: Node functions (entry, planner, continuation checks)
- **`llm_factory.py`**: Multi-provider model creation
- **`simple_llm_selector/`**: Capability inference + routing
- **`state.py`**: TypedDict state management
- **`task_details.py`**: Pydantic models for TODOs

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

**Test Stats**: 🧪 180 test cases, 1793 lines of test code, covering edge cases like special characters, long inputs, and model resolution logic.

---

## 📊 Configuration

All settings via environment variables (`.env` file):

```bash
# Required 🔐
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...

# Optional (default models)
SUB_TASK_MODEL=gpt-4o-mini
SUMMARIZER_MODEL=gemini-2.5-flash

# Optional (temperature tuning)
SUB_TASK_TEMPERATURE=0.0
SUMMARIZER_TEMPERATURE=0.0
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

- **Linear Flow Only**: The graph is currently START→planner→END. We have data structures for fan-out/fan-in (multi-agent evaluation) but it's not implemented yet. Watch this space! 👀
- **No Retry Logic**: Basic `retry_count` tracking but no exponential backoff yet
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
