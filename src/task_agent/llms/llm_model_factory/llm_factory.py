"""
LLM Factory using registry + resolver pattern.

Centralizes model creation with clean separation of concerns:
- Registry: Provider → LangChain constructor mapping
- Resolver: Model name → Provider lookup
- Factory: Instantiates model using registry + resolved provider
"""
import os
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatZhipuAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from sarvam import (
      SarvamChat,
  )

from task_agent.config import settings

# Registry: Provider → LangChain constructor
LLM_REGISTRY: dict[str, Type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "google": ChatGoogleGenerativeAI,
    "groq": ChatGroq,
    "ollama": ChatOllama,
    "Zhipu": ChatZhipuAI,
    "sarvam": SarvamChat,
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
        "chatgpt-": "openai",
        "moonshotai/": "groq",
        "gemini-": "google",
        "gemma-3-27b-it": "google",
        "gemma-3-12b-it": "google",
        "llama-3.3-70b": "groq",
        "llama-3.1-8b": "groq",
        "qwen/": "groq",
        "qwen-": "groq",
        "GLM-4.5": "Zhipu",
        "GLM-4.6V": "Zhipu",
        "GLM-4.7-Flash": "Zhipu",
        "glm-": "ollama",  # Must come after specific Zhipu GLM prefixes
        "llama": "ollama",
        "gemma3:27b": "ollama",
        "sarvam-m": "sarvam",
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

    if settings.USE_OLLAMA_CLOUD_URL and provider == "ollama":
        # for docker deployment or machine without ollama desktop installed, this should be used.
        return ChatOllama(
            model=model,
            base_url=settings.OLLAMA_CLOUD_URL,  # Cloud endpoint
            client_kwargs={
                "headers": {"Authorization": "Bearer " + os.getenv("OLLAMA_API_KEY")},
                "timeout": 60.0  # Timeout in seconds
            }
        )
    return constructor(model=model, **kwargs)


def get_model_provider(model_name: str) -> str:
    """
    Get provider name for a given model.
    """
    return resolve_provider(model_name)
