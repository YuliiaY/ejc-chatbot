"""Chroma vector store holding the crawled EJA/EJC website content."""

from functools import lru_cache

from langchain_chroma import Chroma

from ejc_bot.config import get_settings
from ejc_bot.llm import get_embeddings


@lru_cache
def get_vector_store() -> Chroma | None:
    """Return the persisted store, or ``None`` if the crawler has not been run yet."""
    settings = get_settings()
    if not settings.chroma_dir.exists():
        return None
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=get_embeddings())
