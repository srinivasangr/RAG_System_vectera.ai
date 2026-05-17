"""Central, type-safe settings loaded from .env.

Everything else in the codebase imports `settings` from here.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProviderName = Literal["openai", "anthropic", "gemini", "cerebras", "openrouter"]
EmbeddingProviderName = Literal["openai", "gemini", "snowflake_cortex", "local"]

# Resolve .env relative to the repo root (app/) so it works from any cwd
_APP_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _APP_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Snowflake ---
    snowflake_account: str = ""
    snowflake_user: str = ""
    snowflake_password: str = ""
    snowflake_role: str = "ACCOUNTADMIN"
    snowflake_warehouse: str = "RAG_WH"
    snowflake_database: str = "RAG_DB"
    snowflake_schema: str = "RAG_SCHEMA"

    # --- Provider selection ---
    llm_provider: LLMProviderName = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_provider: EmbeddingProviderName = "snowflake_cortex"
    embedding_model: str = "snowflake-arctic-embed-m-v1.5"
    embedding_dim: int = 768

    # --- Provider keys ---
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    cerebras_api_key: str = ""
    openrouter_api_key: str = ""

    # --- Retrieval tuning ---
    retrieval_top_k: int = 8
    retrieval_candidate_k: int = 30
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 100

    # --- Paths ---
    documents_dir: Path = Field(default=Path("../Documents"))

    @property
    def app_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def documents_path(self) -> Path:
        p = self.documents_dir
        return p if p.is_absolute() else (self.app_root / p).resolve()


settings = Settings()
