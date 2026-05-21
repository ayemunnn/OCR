from functools import lru_cache
import json
from pathlib import Path
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = BACKEND_DIR / "papersleuth.db"
DEFAULT_DATABASE_URL = f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"


class Settings(BaseSettings):
    hf_api_key: str | None = None
    hf_model_name: str = "google/gemma-3-27b-it"
    database_url: str = DEFAULT_DATABASE_URL
    secret_key: str = "change-this-secret-key-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    storage_provider: str = "local"
    azure_storage_connection_string: str | None = None
    azure_storage_container_name: str = "papersleuth-documents"
    backend_cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    @field_validator("backend_cors_origins", mode="before")
    @classmethod
    def parse_backend_cors_origins(cls, value):
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []

            if value.startswith("["):
                try:
                    parsed = json.loads(value)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, list):
                    return [str(origin).strip() for origin in parsed if str(origin).strip()]

            return [origin.strip() for origin in value.split(",") if origin.strip()]

        return value

    model_config = SettingsConfigDict(
        env_file=("backend/.env", "../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
