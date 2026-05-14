from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_api_key: str | None = None
    hf_model_name: str = "google/gemma-3-27b-it"
    database_url: str = "sqlite:///./backend/papersleuth.db"

    model_config = SettingsConfigDict(
        env_file=("backend/.env", "../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
