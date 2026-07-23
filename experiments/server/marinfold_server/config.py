"""Runtime configuration for the MarinFold inference service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings populated from environment variables."""

    model_config = SettingsConfigDict(env_prefix="MARINFOLD_SERVER_")

    token: str | None = None
    environment: str = "local"
    service_name: str = "marinfold-inference-server"


def get_settings() -> Settings:
    """Return service settings from the current process environment."""

    return Settings()
