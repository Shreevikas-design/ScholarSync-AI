from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHOLARSYNC_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: Literal["template", "ollama", "groq"] = "template"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    groq_api_key: SecretStr | None = None
    groq_model: str = "openai/gpt-oss-20b"
    embedding_provider: Literal["ollama", "sentence-transformers"] = "ollama"
    sentence_transformer_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_context_tokens: int = Field(default=8192, ge=1024, le=131072)
    max_output_tokens: int = Field(default=1200, ge=128, le=8192)
    request_timeout: float = Field(default=45.0, ge=5, le=180)
    data_dir: Path = Path("data")
    cache_ttl_hours: int = Field(default=24, ge=0, le=720)
    max_pdf_mb: int = Field(default=25, ge=1, le=100)
    max_pdf_pages: int = Field(default=200, ge=1, le=1000)
    max_retries: int = Field(default=2, ge=0, le=5)
    user_agent: str = "ScholarSyncAI/1.0 (academic research tool)"

    @model_validator(mode="after")
    def validate_token_budgets(self):
        if self.max_output_tokens >= self.max_context_tokens:
            raise ValueError("max_output_tokens must be less than max_context_tokens")
        return self

    @field_validator("ollama_base_url")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        return value.rstrip("/")

    def ensure_data_dirs(self) -> None:
        for name in ("cache", "papers", "chroma", "reports"):
            (self.data_dir / name).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
