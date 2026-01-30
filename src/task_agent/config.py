"""Configuration management for flow-agent using environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Model Configurations
    SUB_TASK_MODEL: str = "gpt-4o-mini"
    SUMMARIZER_MODEL: str = "gemini-2.0-flash-exp"
    DEFAULT_MODEL: str = "gemini-2.0-flash-exp"
    INFERENCE_MODEL: str = "kimi-k2.5:cloud"
    MODERATION_API_CHECK_REQ: bool = True

    # Optional: Model parameters
    SUB_TASK_TEMPERATURE: float = 0.0
    SUMMARIZER_TEMPERATURE: float = 0.0


# Global settings instance
settings = Settings()
