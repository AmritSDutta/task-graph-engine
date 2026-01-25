"""Unit tests for configuration management."""

import os
import pytest
from pydantic import ValidationError

# Import Settings class (env vars are set in conftest.py)
from task_agent.config import Settings


class TestSettingsWithMockedEnv:
    """Tests for Settings with mocked environment variables."""

    def test_settings_requires_openai_api_key(self, monkeypatch):
        """Settings should fail without OPENAI_API_KEY."""
        # Note: The module is already loaded with the global settings instance
        # This test verifies that creating a new Settings() instance fails without required keys

        # Save original values
        original_openai = os.environ.get("OPENAI_API_KEY")
        original_google = os.environ.get("GOOGLE_API_KEY")

        try:
            # Delete both required keys
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]

            # Should raise ValidationError for missing required fields
            with pytest.raises(ValidationError) as exc_info:
                Settings()

            # Just verify it's a validation error about missing fields
            error_str = str(exc_info.value).lower()
            assert "field" in error_str or "required" in error_str
        finally:
            # Restore original values
            if original_openai is not None:
                os.environ["OPENAI_API_KEY"] = original_openai
            if original_google is not None:
                os.environ["GOOGLE_API_KEY"] = original_google

    def test_settings_default_values(self, monkeypatch):
        """Test that Settings have correct default values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")

        settings = Settings()
        assert settings.SUB_TASK_MODEL == "gpt-4o-mini"
        assert settings.SUMMARIZER_MODEL == "gemini-2.0-flash-exp"
        assert settings.DEFAULT_MODEL == "gemini-2.0-flash-exp"
        assert settings.SUB_TASK_TEMPERATURE == 0.0
        assert settings.SUMMARIZER_TEMPERATURE == 0.0
        assert settings.ANTHROPIC_API_KEY == ""

    def test_settings_with_custom_env_vars(self, monkeypatch):
        """Test that Settings can be customized via environment variables."""
        monkeypatch.setenv("OPENAI_API_KEY", "custom-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "custom-google")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "custom-anthropic")
        monkeypatch.setenv("SUB_TASK_MODEL", "gpt-4o")
        monkeypatch.setenv("SUMMARIZER_MODEL", "gemini-2.5-flash")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5-mini")
        monkeypatch.setenv("SUB_TASK_TEMPERATURE", "0.7")
        monkeypatch.setenv("SUMMARIZER_TEMPERATURE", "0.5")

        settings = Settings()
        assert settings.OPENAI_API_KEY == "custom-openai"
        assert settings.GOOGLE_API_KEY == "custom-google"
        assert settings.ANTHROPIC_API_KEY == "custom-anthropic"
        assert settings.SUB_TASK_MODEL == "gpt-4o"
        assert settings.SUMMARIZER_MODEL == "gemini-2.5-flash"
        assert settings.DEFAULT_MODEL == "gpt-5-mini"
        assert settings.SUB_TASK_TEMPERATURE == 0.7
        assert settings.SUMMARIZER_TEMPERATURE == 0.5

    def test_settings_anthropic_api_key_optional(self, monkeypatch):
        """ANTHROPIC_API_KEY should be optional with empty default."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        # Don't set ANTHROPIC_API_KEY

        settings = Settings()
        assert settings.ANTHROPIC_API_KEY == ""

    def test_settings_temperature_must_be_numeric(self, monkeypatch):
        """Temperature values must be numeric (float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("SUB_TASK_TEMPERATURE", "not-a-number")

        with pytest.raises(ValidationError):
            Settings()

    def test_settings_model_names_are_strings(self, monkeypatch):
        """Model configuration values should be strings."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("SUB_TASK_MODEL", "gpt-4o-mini")
        monkeypatch.setenv("SUMMARIZER_MODEL", "gemini-2.5-flash")

        settings = Settings()
        assert isinstance(settings.SUB_TASK_MODEL, str)
        assert isinstance(settings.SUMMARIZER_MODEL, str)
        assert isinstance(settings.DEFAULT_MODEL, str)


class TestSettingsConfiguration:
    """Tests for Settings model configuration."""

    def test_settings_uses_env_file(self):
        """Settings should be configured to use .env file."""
        # Settings already imported at module level

        # Check model_config has correct settings
        assert Settings.model_config["env_file"] == ".env"
        assert Settings.model_config["env_file_encoding"] == "utf-8"
        assert Settings.model_config["case_sensitive"] is True
        assert Settings.model_config["extra"] == "ignore"

    def test_settings_extra_fields_are_ignored(self, monkeypatch):
        """Extra environment variables should be ignored (extra='ignore')."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("UNKNOWN_VARIABLE", "should-be-ignored")

        from importlib import reload
        from task_agent import config

        # Should not raise ValidationError
        reload(config)

        # Unknown variable should not be accessible
        assert not hasattr(config.settings, "UNKNOWN_VARIABLE")

    def test_settings_case_sensitivity(self, monkeypatch):
        """Settings case sensitivity behavior."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        # Note: On Windows, env vars are case-insensitive by default
        # On Unix, case_sensitive=True would cause lowercase to not match
        monkeypatch.setenv("SUB_TASK_MODEL", "custom-model")  # uppercase

        # Settings already imported at module level

        settings = Settings()
        # Uppercase should match
        assert settings.SUB_TASK_MODEL == "custom-model"


class TestSettingsFieldTypes:
    """Tests for Settings field type validation."""

    def test_api_keys_are_strings(self, monkeypatch):
        """API key fields should be strings."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test123")
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test123")

        # Settings already imported at module level

        settings = Settings()
        assert isinstance(settings.OPENAI_API_KEY, str)
        assert isinstance(settings.GOOGLE_API_KEY, str)

    def test_temperature_default_is_float(self, monkeypatch):
        """Temperature default should be float."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        # Settings already imported at module level

        settings = Settings()
        assert isinstance(settings.SUB_TASK_TEMPERATURE, float)
        assert isinstance(settings.SUMMARIZER_TEMPERATURE, float)

    def test_temperature_accepts_integer(self, monkeypatch):
        """Temperature should accept integer values (converted to float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("SUB_TASK_TEMPERATURE", "1")  # string integer

        # Settings already imported at module level

        settings = Settings()
        assert settings.SUB_TASK_TEMPERATURE == 1.0
        assert isinstance(settings.SUB_TASK_TEMPERATURE, float)


class TestSettingsGlobalInstance:
    """Tests for the global settings instance."""

    def test_settings_instance_exists(self, monkeypatch):
        """The global settings instance should exist."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        from task_agent import config

        assert config.settings is not None
        assert isinstance(config.settings, config.Settings)

    def test_settings_is_singleton_like(self, monkeypatch):
        """The settings instance should be consistent."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        from task_agent import config

        settings1 = config.settings
        settings2 = config.settings
        assert settings1 is settings2


class TestSettingsWithEnvFile:
    """Tests for .env file integration."""

    def test_env_file_path_is_dotenv(self):
        """Settings should be configured to read from .env file."""
        # Settings already imported at module level
        assert Settings.model_config["env_file"] == ".env"

    def test_env_file_encoding_is_utf8(self):
        """Settings should use UTF-8 encoding for .env file."""
        # Settings already imported at module level
        assert Settings.model_config["env_file_encoding"] == "utf-8"


class TestSettingsRequiredFields:
    """Tests for required field validation."""

    @pytest.mark.parametrize("field", ["OPENAI_API_KEY", "GOOGLE_API_KEY"])
    def test_required_fields_validation(self, field, monkeypatch):
        """Test that required fields raise ValidationError when missing."""
        # Set all required fields except the one being tested
        required_fields = {
            "OPENAI_API_KEY": "test-openai",
            "GOOGLE_API_KEY": "test-google",
        }
        for key in required_fields:
            if key != field:
                monkeypatch.setenv(key, required_fields[key])
            else:
                monkeypatch.delenv(key, raising=False)

        # Settings already imported at module level

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        # Check that the missing field is mentioned in error
        error_str = str(exc_info.value).lower()
        assert field.lower() in error_str or "field required" in error_str


class TestSettingsDefaults:
    """Tests for default values."""

    def test_default_model_names(self, monkeypatch):
        """Test default model names are set correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        # Settings already imported at module level

        settings = Settings()
        # Model defaults
        assert settings.SUB_TASK_MODEL == "gpt-4o-mini"
        assert settings.SUMMARIZER_MODEL == "gemini-2.0-flash-exp"
        assert settings.DEFAULT_MODEL == "gemini-2.0-flash-exp"

    def test_default_temperature_values(self, monkeypatch):
        """Test default temperature values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        # Settings already imported at module level

        settings = Settings()
        assert settings.SUB_TASK_TEMPERATURE == 0.0
        assert settings.SUMMARIZER_TEMPERATURE == 0.0


class TestSettingsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_api_key_is_accepted(self, monkeypatch):
        """Empty string for API key is accepted (Pydantic default behavior)."""
        monkeypatch.setenv("OPENAI_API_KEY", "")  # Empty string
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        # Settings already imported at module level

        # Empty string is valid for str type in Pydantic (though not ideal)
        settings = Settings()
        assert settings.OPENAI_API_KEY == ""

    def test_whitespace_api_key_is_preserved(self, monkeypatch):
        """Whitespace in API keys should be preserved as-is."""
        monkeypatch.setenv("OPENAI_API_KEY", " sk-test-with-space ")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        from importlib import reload
        from task_agent import config

        reload(config)

        assert config.settings.OPENAI_API_KEY == " sk-test-with-space "

    def test_temperature_negative_value(self, monkeypatch):
        """Negative temperature values should be accepted (it's a valid float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("SUB_TASK_TEMPERATURE", "-0.5")

        from importlib import reload
        from task_agent import config

        reload(config)

        assert config.settings.SUB_TASK_TEMPERATURE == -0.5

    def test_temperature_high_value(self, monkeypatch):
        """High temperature values should be accepted (it's a valid float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("SUB_TASK_TEMPERATURE", "2.5")

        from importlib import reload
        from task_agent import config

        reload(config)

        assert config.settings.SUB_TASK_TEMPERATURE == 2.5
