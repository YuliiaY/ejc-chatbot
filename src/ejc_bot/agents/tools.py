"""Tools the visitor agent can call."""

from langchain_core.tools import tool

from ejc_bot.config import get_settings
from ejc_bot.stores.db import run_select
from ejc_bot.stores.vector import get_vector_store


@tool
def database_rag_search(query: str) -> str:
    """Search internal database for general information about EJA and EJC
    (history, FAQs, rules, tickets)."""
    vector_store = get_vector_store()
    if vector_store is None:
        return "Error: vector store not found. Please run `ejc-ingest-web` first."

    retriever = vector_store.as_retriever(search_kwargs={"k": get_settings().retriever_k})
    matching_docs = retriever.invoke(query)
    if not matching_docs:
        return "No relevant information found in vector storage."

    context_blocks = []
    for doc in matching_docs:
        source_url = doc.metadata.get("source", "Unknown source")
        page_title = doc.metadata.get("title", source_url)
        context_blocks.append(
            f"[Source Title: {page_title}]\n[Source URL: {source_url}]\n"
            f"Content:\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_blocks)


@tool
def query_event_schedule_sql(sql_query: str) -> str:
    """Execute a read-only SQLite SELECT query against the EJC event schedule table.
    Table name: events
    Columns: id, title, day, start_time, end_time, location, description, source_url
    The `day` column holds text like 'Tuesday, August 4th'; times are 'HH:MM' strings.
    Example query:
        SELECT title, start_time, location FROM events
        WHERE day LIKE 'Tuesday%' AND title LIKE '%show%'
    """
    try:
        rows = run_select(get_settings().sql_db_path, sql_query)
    except Exception as e:
        return f"SQL Execution Error: {e}"

    if not rows:
        return "No schedule results found matching that SQL query."
    return str(rows)


VISITOR_TOOLS = [database_rag_search, query_event_schedule_sql]
