"""Central configuration.

Every value can be overridden with an ``EJC_``-prefixed environment variable or in ``.env``,
e.g. ``EJC_LLM_PROVIDER=openai`` + ``EJC_LLM_BASE_URL=http://localhost:8000/v1`` to switch
from Ollama to a vLLM server without touching code.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["ollama", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="EJC_", extra="ignore")

    # --- Convention ---
    convention_year: int = 2026
    seed_urls: list[str] = ["https://eja.net/", "https://ejc2026.org/"]
    program_url: str = "https://ejc2026.org/program/"

    # --- Storage ---
    data_dir: Path = Path("data")

    # --- Chat model ---
    # "ollama" uses Ollama's native API; "openai" talks to any OpenAI-compatible server (vLLM, ...).
    llm_provider: Provider = "ollama"
    llm_model: str = "llama3.1"
    llm_base_url: str | None = None
    llm_api_key: SecretStr = SecretStr("not-needed")
    llm_temperature: float = 0.0

    # --- Embeddings ---
    embedding_provider: Provider = "ollama"
    embedding_model: str = "nomic-embed-text"
    embedding_base_url: str | None = None

    # --- Retrieval ---
    retriever_k: int = 4

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma_db"

    @property
    def sql_db_path(self) -> Path:
        return self.data_dir / "schedule.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
