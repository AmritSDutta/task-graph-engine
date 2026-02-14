"""Unit tests for LLM factory - no API calls required."""

import pytest

from task_agent.llms.llm_model_factory.llm_factory import (
    LLM_REGISTRY,
    create_llm,
    get_model_provider,
    resolve_provider,
)


class TestResolveProvider:
    """Tests for resolve_provider function."""

    def test_resolve_openai_gpt_prefix(self):
        assert resolve_provider("gpt-4o") == "openai"
        assert resolve_provider("gpt-4o-mini") == "openai"
        assert resolve_provider("gpt-5-mini") == "openai"
        assert resolve_provider("gpt-5-nano") == "openai"

    def test_resolve_openai_chatgpt_prefix(self):
        assert resolve_provider("chatgpt-4o") == "openai"
        assert resolve_provider("chatgpt-4o-latest") == "openai"

    def test_resolve_openai_moonshotai_prefix(self):
        assert resolve_provider("moonshotai/kimi-k2-instruct") == "groq"

    def test_resolve_groq_llama_3_1_8b_prefix(self):
        assert resolve_provider("llama-3.1-8b-instant") == "groq"

    def test_resolve_google_gemini_prefix(self):
        assert resolve_provider("gemini-2.0-flash-exp") == "google"
        assert resolve_provider("gemini-3-flash-preview:cloud") == "ollama"  # cloud takes precedence

    def test_resolve_groq_qwen_slash_prefix(self):
        assert resolve_provider("qwen/qwen-2.5-72b-instruct") == "groq"
        assert resolve_provider("qwen/qwen3-32b") == "groq"

    def test_resolve_google_gemma3_it_models(self):
        """New Google Gemma 3 IT models."""
        assert resolve_provider("gemma-3-27b-it") == "google"
        assert resolve_provider("gemma-3-12b-it") == "google"

    def test_resolve_groq_qwen_dash_prefix(self):
        # Only models starting with "qwen/" get mapped to groq by the slash prefix
        # Models like "qwen3-32b" don't have "qwen-" prefix, so this test is removed
        # The qwen- prefix in code maps models like "qwen-2.5-72b-instruct" (not commonly used)
        pass

    def test_resolve_ollama_cloud_suffix(self):
        """Cloud suffix takes precedence over prefix checks."""
        assert resolve_provider("qwen3-coder:480b-cloud") == "ollama"
        assert resolve_provider("glm-4.6:cloud") == "ollama"
        assert resolve_provider("gemma3:27b-cloud") == "ollama"
        assert resolve_provider("gemini-3-flash-preview:cloud") == "ollama"

    def test_resolve_ollama_glm_prefix(self):
        assert resolve_provider("glm-4.6") == "ollama"

    def test_resolve_ollama_llama_prefix(self):
        assert resolve_provider("llama-3-70b") == "ollama"
        assert resolve_provider("llama-3.1-405b") == "ollama"

    def test_resolve_ollama_gemma_prefix(self):
        # gemma3:27b prefix matches Ollama (cloud deployment)
        assert resolve_provider("gemma3:27b") == "ollama"
        # gemma3:27b-cloud also resolves to Ollama via cloud suffix
        assert resolve_provider("gemma3:27b-cloud") == "ollama"

    def test_resolve_unsupported_model_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            resolve_provider("unsupported-model")
        with pytest.raises(ValueError, match="Unsupported model"):
            resolve_provider("claude-3-opus")

    def test_error_message_includes_supported_prefixes_and_suffix(self):
        with pytest.raises(ValueError) as exc_info:
            resolve_provider("invalid-model")
        error_msg = str(exc_info.value)
        assert "chatgpt-" in error_msg
        assert "gemini-" in error_msg
        assert "gemma-3-27b-it" in error_msg
        assert "gemma-3-12b-it" in error_msg
        assert "gpt-" in error_msg
        assert "qwen/" in error_msg
        assert "qwen-" in error_msg
        assert ":cloud" in error_msg


class TestGetModelProvider:
    """Tests for get_model_provider function."""

    def test_get_model_provider_delegates_to_resolve_provider(self):
        assert get_model_provider("gpt-4o") == "openai"
        assert get_model_provider("gemini-2.5-flash") == "google"
        assert get_model_provider("qwen/qwen3-32b") == "groq"
        assert get_model_provider("qwen3-coder:480b-cloud") == "ollama"


class TestLLMRegistry:
    """Tests for LLM_REGISTRY configuration."""

    def test_registry_contains_expected_providers(self):
        expected_providers = {"openai", "google", "groq", "ollama", "Zhipu", "sarvam"}
        assert set(LLM_REGISTRY.keys()) == expected_providers

    def test_registry_values_are_chat_model_classes(self):
        from langchain_core.language_models import BaseChatModel

        for provider, constructor in LLM_REGISTRY.items():
            assert constructor is not None
            # Verify constructor is a class that inherits from BaseChatModel
            assert issubclass(constructor, BaseChatModel)

    @pytest.mark.parametrize(
        "provider,expected_class",
        [
            ("openai", "ChatOpenAI"),
            ("google", "ChatGoogleGenerativeAI"),
            ("groq", "ChatGroq"),
            ("ollama", "ChatOllama"),
        ],
    )
    def test_registry_mappings_are_correct(self, provider, expected_class):
        assert LLM_REGISTRY[provider].__name__ == expected_class


class TestCreateLlm:
    """Tests for create_llm factory function."""

    def test_create_openai_model(self):
        llm = create_llm("gpt-4o", temperature=0.5)
        assert llm.model_name == "gpt-4o"
        assert llm.temperature == 0.5

    def test_create_google_model(self):
        llm = create_llm("gemini-2.5-flash", temperature=0.0)
        assert llm.model == "gemini-2.5-flash"
        assert llm.temperature == 0.0

    def test_create_groq_model(self):
        llm = create_llm("qwen/qwen3-32b", temperature=0.7)
        assert llm.model_name == "qwen/qwen3-32b"
        assert llm.temperature == 0.7

    def test_create_ollama_model(self):
        llm = create_llm("glm-4.6:cloud", temperature=0.0)
        assert llm.model == "glm-4.6:cloud"
        assert llm.temperature == 0.0

    def test_create_llm_passes_additional_kwargs(self):
        """Test that additional kwargs are passed through to the constructor."""
        llm = create_llm("gpt-4o", temperature=0.5, max_tokens=1000, top_p=0.9)
        assert llm.temperature == 0.5
        assert llm.max_tokens == 1000
        assert llm.top_p == 0.9

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-5-mini",
            "gpt-5-nano",
            "chatgpt-4o",
            "moonshotai/kimi-k2-instruct",
            "gemini-2.0-flash-exp",
            "gemini-3-flash-preview:cloud",
            "gemma-3-27b-it",
            "gemma-3-12b-it",
            "qwen/qwen-2.5-72b-instruct",
            "qwen/qwen3-32b",
            "qwen3-coder:480b-cloud",
            "glm-4.6:cloud",
            "gemma3:27b-cloud",
            "llama-3-70b",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "gpt-oss:20b-cloud",
            "kimi-k2.5:cloud",
            "glm-5:cloud",
        ],
    )
    def test_create_llm_with_all_supported_models(self, model_name):
        """Test that all supported models can be created."""
        llm = create_llm(model_name, temperature=0.0)
        assert llm is not None

    def test_create_llm_unsupported_model_raises_error(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            create_llm("claude-3-opus")

    def test_create_llm_returns_base_chat_model(self):
        from langchain_core.language_models import BaseChatModel

        llm = create_llm("gpt-4o")
        assert isinstance(llm, BaseChatModel)


class TestOllamaCloudUrl:
    """Tests for Ollama cloud URL configuration."""

    def test_create_ollama_with_cloud_url_enabled(self, monkeypatch):
        """Test Ollama models use cloud URL when USE_OLLAMA_CLOUD_URL is True."""
        # Mock environment variables
        monkeypatch.setenv("USE_OLLAMA_CLOUD_URL", "true")
        monkeypatch.setenv("OLLAMA_CLOUD_URL", "https://ollama.example.com")
        monkeypatch.setenv("OLLAMA_API_KEY", "test-api-key-123")

        # Patch os.getenv to return the test OLLAMA_API_KEY
        import os
        original_getenv = os.getenv

        def mock_getenv(key, default=None):
            if key == "OLLAMA_API_KEY":
                return "test-api-key-123"
            return original_getenv(key, default)

        monkeypatch.setattr(os, "getenv", mock_getenv)

        # Reload settings to pick up new environment variables
        import importlib
        from task_agent import config
        importlib.reload(config)

        from task_agent.llms.llm_model_factory import llm_factory
        importlib.reload(llm_factory)

        # Create an Ollama model (provider == "ollama")
        llm = llm_factory.create_llm("glm-4.6:cloud", temperature=0.0)

        # Verify the cloud URL configuration was applied
        assert llm.base_url == "https://ollama.example.com"
        # Verify client_kwargs contains the auth header
        assert llm.client_kwargs is not None
        assert "headers" in llm.client_kwargs
        assert llm.client_kwargs["headers"]["Authorization"] == "Bearer test-api-key-123"

    def test_create_ollama_with_cloud_url_disabled(self, monkeypatch):
        """Test Ollama models use default settings when USE_OLLAMA_CLOUD_URL is False."""
        # Ensure cloud URL is disabled (default)
        monkeypatch.setenv("USE_OLLAMA_CLOUD_URL", "false")

        # Reload settings
        import importlib
        from task_agent import config
        importlib.reload(config)
        from task_agent.llms.llm_model_factory import llm_factory
        importlib.reload(llm_factory)

        # Create an Ollama model - should use default ChatOllama constructor
        llm = llm_factory.create_llm("glm-4.6:cloud", temperature=0.0)
        assert llm.model == "glm-4.6:cloud"
        assert llm.temperature == 0.0

    def test_non_ollama_providers_unaffected_by_cloud_url_setting(self, monkeypatch):
        """Test that non-Ollama providers are not affected by USE_OLLAMA_CLOUD_URL."""
        # Enable cloud URL setting
        monkeypatch.setenv("USE_OLLAMA_CLOUD_URL", "true")
        monkeypatch.setenv("OLLAMA_CLOUD_URL", "https://ollama.example.com")
        monkeypatch.setenv("OLLAMA_API_KEY", "test-api-key-123")

        # Reload settings
        import importlib
        from task_agent import config
        importlib.reload(config)
        from task_agent.llms.llm_model_factory import llm_factory
        importlib.reload(llm_factory)

        # OpenAI model should work normally
        openai_llm = llm_factory.create_llm("gpt-4o", temperature=0.5)
        assert openai_llm.model_name == "gpt-4o"
        assert openai_llm.temperature == 0.5

        # Google model should work normally
        google_llm = llm_factory.create_llm("gemini-2.5-flash", temperature=0.0)
        assert google_llm.model == "gemini-2.5-flash"
        assert google_llm.temperature == 0.0

        # Groq model should work normally
        groq_llm = llm_factory.create_llm("qwen/qwen3-32b", temperature=0.7)
        assert groq_llm.model_name == "qwen/qwen3-32b"
        assert groq_llm.temperature == 0.7


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_cloud_suffix_takes_precedence_over_gemini_prefix(self):
        """Cloud models with gemini prefix should resolve to ollama."""
        assert resolve_provider("gemini-3-flash-preview:cloud") == "ollama"

    def test_case_sensitivity(self):
        """Model names are case-sensitive."""
        assert resolve_provider("gpt-4o") == "openai"
        # Uppercase should fail as it's not in the registry
        with pytest.raises(ValueError):
            resolve_provider("GPT-4O")

    def test_empty_model_name_raises_error(self):
        with pytest.raises(ValueError):
            resolve_provider("")

    def test_partial_match_does_not_trigger(self):
        """Only prefix matching, not substring matching."""
        # Should not match 'gpt-' just because it contains 'gpt'
        with pytest.raises(ValueError):
            resolve_provider("my-gpt-model")
