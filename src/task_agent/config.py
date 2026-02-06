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
    INFERENCE_MODEL: str = "sarvam-m"  # other option : gpt-5-nano
    FALLBACK_MODEL: str = "gpt-4o-mini"
    INFERENCE_MAX_RETRY: int = 3
    MODERATION_API_CHECK_REQ: bool = True
    COST_SPREADING_FACTOR: float = 0.03
    TOKEN_USAGE_LOG_BASE: float = 100.0
    ENABLE_LANGSMITH_TRACING_V2: str = "false"
    USE_OLLAMA_CLOUD_URL: bool = False
    OLLAMA_CLOUD_URL: str = "https://ollama.com"

    MODEL_COST_CSV_PATH: str = "model_costs.csv"
    """Path to model costs CSV file.
    Resolution strategy:
    - Absolute: /app/config/model_costs.csv (e.g., Docker mount)
    - Relative: ./config/model_costs.csv (from current working directory)
    - Filename: model_costs.csv (from project root)

    Example (Docker):
        docker run -e MODEL_COST_CSV_PATH=/app/config/costs.csv \\
          -v /host/config:/app/config task-graph-engine
    """

    MODEL_CAPABILITY_CSV_PATH: str = "model_capabilities.csv"
    """Path to model capabilities CSV file.
    Resolution strategy:
    - Absolute: /app/config/model_capabilities.csv (e.g., Docker mount)
    - Relative: ./config/model_capabilities.csv (from current working directory)
    - Filename: model_capabilities.csv (from project root)

    Example (Docker):
        docker run -e MODEL_CAPABILITY_CSV_PATH=/app/config/caps.csv \\
          -v /host/config:/app/config task-graph-engine
    """

    # API Authentication
    API_KEY: str = ""
    """API key for protected endpoints.
    If empty and REQUIRE_AUTH=true, all requests will be rejected.
    Set via API_KEY env var."""

    REQUIRE_AUTH: bool = False
    """Whether to require authentication for protected endpoints.
    Set via REQUIRE_AUTH=true env var."""


# Global settings instance
settings = Settings()
