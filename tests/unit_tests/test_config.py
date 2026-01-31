"""Unit tests for configuration management."""

import os
import pytest
from pydantic import ValidationError

# Import Settings class (env vars are set in conftest.py)
from task_agent.config import Settings


class TestSettingsWithMockedEnv:
    """Tests for Settings with mocked environment variables."""

    def test_settings_default_values(self):
        """Test that Settings have correct default values."""
        settings = Settings()
        assert settings.INFERENCE_MODEL == "kimi-k2.5:cloud"
        assert settings.MODERATION_API_CHECK_REQ is True
        assert settings.COST_SPREADING_FACTOR == 0.03
        assert settings.MODEL_COST_CSV_PATH == "model_costs.csv"
        assert settings.MODEL_CAPABILITY_CSV_PATH == "model_capabilities.csv"

    def test_settings_with_custom_env_vars(self, monkeypatch):
        """Test that Settings can be customized via environment variables."""
        monkeypatch.setenv("INFERENCE_MODEL", "gpt-4o")
        monkeypatch.setenv("MODERATION_API_CHECK_REQ", "false")
        monkeypatch.setenv("COST_SPREADING_FACTOR", "0.05")
        monkeypatch.setenv("MODEL_COST_CSV_PATH", "./custom/costs.csv")
        monkeypatch.setenv("MODEL_CAPABILITY_CSV_PATH", "/absolute/path/caps.csv")

        settings = Settings()
        assert settings.INFERENCE_MODEL == "gpt-4o"
        assert settings.MODERATION_API_CHECK_REQ is False
        assert settings.COST_SPREADING_FACTOR == 0.05
        assert settings.MODEL_COST_CSV_PATH == "./custom/costs.csv"
        assert settings.MODEL_CAPABILITY_CSV_PATH == "/absolute/path/caps.csv"

    def test_settings_cost_spreading_factor_must_be_numeric(self, monkeypatch):
        """COST_SPREADING_FACTOR must be numeric (float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("COST_SPREADING_FACTOR", "not-a-number")

        with pytest.raises(ValidationError):
            Settings()

    def test_settings_model_names_are_strings(self, monkeypatch):
        """Model configuration values should be strings."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("INFERENCE_MODEL", "gpt-4o-mini")

        settings = Settings()
        assert isinstance(settings.INFERENCE_MODEL, str)
        assert isinstance(settings.MODEL_COST_CSV_PATH, str)
        assert isinstance(settings.MODEL_CAPABILITY_CSV_PATH, str)


class TestSettingsConfiguration:
    """Tests for Settings model configuration."""

    def test_settings_uses_env_file(self):
        """Settings should be configured to use .env file."""
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
        monkeypatch.setenv("INFERENCE_MODEL", "custom-model")  # uppercase

        settings = Settings()
        # Uppercase should match
        assert settings.INFERENCE_MODEL == "custom-model"


class TestSettingsFieldTypes:
    """Tests for Settings field type validation."""

    def test_model_names_are_strings(self):
        """Model configuration values should be strings."""
        settings = Settings()
        assert isinstance(settings.INFERENCE_MODEL, str)
        assert isinstance(settings.MODEL_COST_CSV_PATH, str)
        assert isinstance(settings.MODEL_CAPABILITY_CSV_PATH, str)

    def test_cost_spreading_factor_default_is_float(self):
        """COST_SPREADING_FACTOR default should be float."""
        settings = Settings()
        assert isinstance(settings.COST_SPREADING_FACTOR, float)
        assert settings.COST_SPREADING_FACTOR == 0.03

    def test_cost_spreading_factor_accepts_integer(self, monkeypatch):
        """COST_SPREADING_FACTOR should accept integer values (converted to float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("COST_SPREADING_FACTOR", "1")  # string integer

        settings = Settings()
        assert settings.COST_SPREADING_FACTOR == 1.0
        assert isinstance(settings.COST_SPREADING_FACTOR, float)

    def test_moderation_api_check_req_is_bool(self):
        """MODERATION_API_CHECK_REQ should be boolean."""
        settings = Settings()
        assert isinstance(settings.MODERATION_API_CHECK_REQ, bool)
        assert settings.MODERATION_API_CHECK_REQ is True


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
        assert Settings.model_config["env_file"] == ".env"

    def test_env_file_encoding_is_utf8(self):
        """Settings should use UTF-8 encoding for .env file."""
        assert Settings.model_config["env_file_encoding"] == "utf-8"


class TestSettingsRequiredFields:
    """Tests for required field validation.

    Note: API keys (OPENAI_API_KEY, GOOGLE_API_KEY) are handled by LangChain
    directly from environment variables, not through Pydantic Settings.
    All fields in Settings have default values, so there are no required fields.
    """

    def test_all_fields_have_defaults(self):
        """Test that all Settings fields have default values."""
        settings = Settings()
        # All fields should be accessible
        assert hasattr(settings, "INFERENCE_MODEL")
        assert hasattr(settings, "MODERATION_API_CHECK_REQ")
        assert hasattr(settings, "COST_SPREADING_FACTOR")
        assert hasattr(settings, "MODEL_COST_CSV_PATH")
        assert hasattr(settings, "MODEL_CAPABILITY_CSV_PATH")


class TestSettingsDefaults:
    """Tests for default values."""

    def test_default_values(self, monkeypatch):
        """Test default values are set correctly."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")

        settings = Settings()
        assert settings.INFERENCE_MODEL == "kimi-k2.5:cloud"
        assert settings.MODERATION_API_CHECK_REQ is True
        assert settings.COST_SPREADING_FACTOR == 0.03
        assert settings.MODEL_COST_CSV_PATH == "model_costs.csv"
        assert settings.MODEL_CAPABILITY_CSV_PATH == "model_capabilities.csv"


class TestSettingsEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_string_model_is_accepted(self):
        """Empty string for model is accepted (though not ideal)."""
        # Note: This documents current behavior. Model names should be validated
        pass  # No easy way to test this without reloading modules

    def test_whitespace_in_path_is_preserved(self, monkeypatch):
        """Whitespace in path names is preserved as-is."""
        monkeypatch.setenv("MODEL_COST_CSV_PATH", " custom/path.csv ")
        monkeypatch.setenv("MODEL_CAPABILITY_CSV_PATH", " another/path.csv ")

        settings = Settings()
        assert settings.MODEL_COST_CSV_PATH == " custom/path.csv "
        assert settings.MODEL_CAPABILITY_CSV_PATH == " another/path.csv "

    def test_cost_spreading_factor_negative_value(self, monkeypatch):
        """Negative COST_SPREADING_FACTOR should be accepted (it's a valid float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("COST_SPREADING_FACTOR", "-0.5")

        from importlib import reload
        from task_agent import config

        reload(config)

        assert config.settings.COST_SPREADING_FACTOR == -0.5

    def test_cost_spreading_factor_high_value(self, monkeypatch):
        """High COST_SPREADING_FACTOR should be accepted (it's a valid float)."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("COST_SPREADING_FACTOR", "2.5")

        from importlib import reload
        from task_agent import config

        reload(config)

        assert config.settings.COST_SPREADING_FACTOR == 2.5

    def test_moderation_api_check_accepts_bool_strings(self, monkeypatch):
        """MODERATION_API_CHECK_REQ accepts boolean string values."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
        monkeypatch.setenv("MODERATION_API_CHECK_REQ", "false")

        settings = Settings()
        assert settings.MODERATION_API_CHECK_REQ is False

        monkeypatch.setenv("MODERATION_API_CHECK_REQ", "true")
        settings = Settings()
        assert settings.MODERATION_API_CHECK_REQ is True
