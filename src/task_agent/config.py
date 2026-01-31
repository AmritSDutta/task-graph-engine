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
    INFERENCE_MODEL: str = "kimi-k2.5:cloud"
    MODERATION_API_CHECK_REQ: bool = True
    COST_SPREADING_FACTOR: float = 0.03

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


# Global settings instance
settings = Settings()
