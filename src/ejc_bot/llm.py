"""Model factories, so the rest of the code never depends on a specific inference backend."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ejc_bot.config import Settings, get_settings


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    s = settings or get_settings()
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        extra = {"base_url": s.llm_base_url} if s.llm_base_url else {}
        return ChatOllama(model=s.llm_model, temperature=s.llm_temperature, **extra)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        temperature=s.llm_temperature,
    )


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    s = settings or get_settings()
    if s.embedding_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        extra = {"base_url": s.embedding_base_url} if s.embedding_base_url else {}
        return OllamaEmbeddings(model=s.embedding_model, **extra)

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=s.embedding_model,
        base_url=s.embedding_base_url,
        api_key=s.llm_api_key,
        check_embedding_ctx_length=False,
    )
