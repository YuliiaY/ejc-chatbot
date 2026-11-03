from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ejc_bot.config import Settings
from ejc_bot.llm import get_chat_model, get_embeddings


def test_defaults_use_ollama():
    settings = Settings()
    assert isinstance(get_chat_model(settings), ChatOllama)
    assert settings.sql_db_path == settings.data_dir / "schedule.db"


def test_openai_compatible_backend_from_env(monkeypatch):
    monkeypatch.setenv("EJC_LLM_PROVIDER", "openai")
    monkeypatch.setenv("EJC_LLM_BASE_URL", "http://vllm:8000/v1")
    monkeypatch.setenv("EJC_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("EJC_EMBEDDING_PROVIDER", "openai")

    settings = Settings()
    model = get_chat_model(settings)

    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert model.openai_api_base == "http://vllm:8000/v1"
    assert isinstance(get_embeddings(settings), OpenAIEmbeddings)
