"""
LLM Factory using registry + resolver pattern.

Centralizes model creation with clean separation of concerns:
- Registry: Provider → LangChain constructor mapping
- Resolver: Model name → Provider lookup
- Factory: Instantiates model using registry + resolved provider
"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# Registry: Provider → LangChain constructor
LLM_REGISTRY: dict[str, Type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "google": ChatGoogleGenerativeAI,
    "groq": ChatGroq,
    "ollama": ChatOllama,
}


def resolve_provider(model: str) -> str:
    """
    Pure resolver: Extract provider from model name suffix/prefix.

    Suffix check happens before prefix check:
    - Models containing :cloud → Ollama (local deployment)
    - Otherwise, uses prefix-based provider resolution


    """
    # Check suffix first (cloud models go to Ollama)
    if "cloud" in model:
        return "ollama"

    # Fallback to prefix-based resolution
    prefixes = {
        "gpt-": "openai",
        "gemini-": "google",
        "chatgpt-": "openai",
        "qwen/": "groq",
        "qwen-": "groq",
        "glm-": "ollama",
        "llama": "ollama",
        "gemma": "ollama",
    }

    for prefix, provider in prefixes.items():
        if model.startswith(prefix):
            return provider

    raise ValueError(
        f"Unsupported model: {model!r}. "
        f"Supported prefixes: {sorted(prefixes.keys())}, "
        f"supported suffix: :cloud"
    )


def create_llm(model: str, **kwargs) -> BaseChatModel:
    """
    Factory: Create LangChain model instance.use create_llm

    Example:
        model = create_llm("gpt-4o", temperature=0.5)
        response = model.invoke("Hello")
    """
    provider = resolve_provider(model)
    constructor = LLM_REGISTRY[provider]
    return constructor(model=model, **kwargs)


def get_model_provider(model_name: str) -> str:
    """
    Get provider name for a given model.
    """
    return resolve_provider(model_name)
