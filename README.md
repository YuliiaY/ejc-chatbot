# EJC Assistant

An LLM assistant for the **European Juggling Convention (EJC)** and the **European Juggling
Association (EJA)**. It answers visitor questions about the convention and cites its sources, using
two kinds of knowledge:

- **Website knowledge (RAG):** eja.net and the EJC site, including linked PDFs. These are crawled,
  split into chunks and embedded into a Chroma vector store.
- **Schedule knowledge (text-to-SQL):** the program page is scraped into SQLite, and the agent
  queries it read-only.

Everything runs on **local, open-weight models** (Ollama by default). 


## Quickstart

```bash
# 1. Install dependencies (Python 3.12, uv)
uv sync

# 2. Start a local model server and pull the models
ollama pull llama3.1 && ollama pull nomic-embed-text

# 3. Build the knowledge bases (writes to ./data)
uv run ejc-ingest-schedule
uv run ejc-ingest-web          # crawls the websites; takes a few minutes

# 4. Chat
uv run chainlit run src/ejc_bot/ui/chainlit_app.py
```

## Configuration

All settings live in [src/ejc_bot/config.py](src/ejc_bot/config.py). You can override them with
`EJC_*` environment variables or in a `.env` file. For example, to use a vLLM server:

```bash
EJC_LLM_PROVIDER=openai
EJC_LLM_BASE_URL=http://localhost:8000/v1
EJC_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

| Variable | Default | Purpose |
|---|---|---|
| `EJC_CONVENTION_YEAR` | `2026` | Convention the bot talks about |
| `EJC_SEED_URLS` | `["https://eja.net/", "https://ejc2026.org/"]` | Crawl start points (JSON list) |
| `EJC_PROGRAM_URL` | `https://ejc2026.org/program/` | Schedule page |
| `EJC_DATA_DIR` | `data` | Location of the vector store and SQLite DB |
| `EJC_LLM_PROVIDER` / `_MODEL` / `_BASE_URL` | `ollama` / `llama3.1` / – | Chat model backend |
| `EJC_EMBEDDING_PROVIDER` / `_MODEL` | `ollama` / `nomic-embed-text` | Embedding backend |

## Development
To run tests:
```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

