import os
import sqlite3

import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent

# --- 1. SETUP LOCAL VECTOR STORE ---
SQL_DB_PATH = "./schedule.db"
DB_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(DB_PATH):
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
else:
    vector_store = None
    retriever = None

# --- 2. DEFINE SYSTEM AGENT TOOLS ---

@tool
def database_rag_search(query: str) -> str:
    """Search internal database for general information about EJA and EJC 2026 (history, FAQs, rules, tickets)."""
    if not retriever:
        return "Error: Database './chroma_db' folder not found. Please run the crawler script first."
    
    matching_docs = retriever.invoke(query)
    if not matching_docs:
        return "No relevant information found in vector storage."
    
    context_blocks = []
    for doc in matching_docs:
        source_url = doc.metadata.get("source", "Unknown source")
        page_title = doc.metadata.get("title", source_url)
        context_blocks.append(
            f"[Source Title: {page_title}]\n[Source URL: {source_url}]\nContent:\n{doc.page_content}"
        )
        
    return "\n\n---\n\n".join(context_blocks)

@tool
def query_event_schedule_sql(sql_query: str) -> str:
    """Execute SQLite read-only queries against the EJC 2026 event schedule table.
    Table name: events
    Columns: id, title, day, date, start_time, end_time, location, description, category, source_url
    Example query: SELECT title, start_time, location FROM events WHERE day = 'Tuesday' OR title LIKE '%show%'
    """
    try:
        conn = sqlite3.connect(SQL_DB_PATH)
        cursor = conn.cursor()
        
        # Safety check: Prevent destructive queries
        clean_query = sql_query.strip().lower()
        if not clean_query.startswith("select"):
            return "Error: Only SELECT queries are permitted on the schedule database."
            
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        
        # Format results for the LLM
        col_names = [description[0] for description in cursor.description]
        conn.close()
        
        if not rows:
            return "No schedule results found matching that SQL query."
            
        results = [dict(zip(col_names, row)) for row in rows]
        return str(results)
    except Exception as e:
        return f"SQL Execution Error: {e}"

tools = [database_rag_search, query_event_schedule_sql]


# --- 3. AGENT SYSTEM PROMPT ---

SYSTEM_PROMPT = """You are a helpful event assistant for EJA and EJC 2026. 
Use the `database_rag_search` tool to look up accurate information about the events.

Tool Usage Guide:
1. For event timing, dates, locations, or schedule lookups, USE `query_event_schedule_sql`.
2. For general questions (accommodations, registration, venue rules), USE `database_rag_search`.

CRITICAL CITATION RULES:
1. Always base your answers strictly on the retrieved context or SQL results.
2. Whenever you present facts retrieved from the database, you MUST include markdown source links at the bottom of your response.
3. If you use `query_event_schedule_sql` to retrieve schedule information, you MUST include the source URL of the schedule page (https://ejc2026.org/program/) in your response.
4. Format links like this: `[Page Title](URL)`.
5. If multiple sources were used, list them cleanly under a "### Sources" header.
"""

# --- 4. LIFECYCLE HOOKS & CHAINLIT RUNTIME ---

@cl.on_chat_start
async def on_chat_start():
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    # Fixed: Passing system instructions via the 'prompt' parameter
    agent_executor = create_react_agent(
        llm, 
        tools, 
        prompt=SYSTEM_PROMPT
    )
    cl.user_session.set("agent", agent_executor)
    
    status_msg = "🤹 **EJA / EJC 2026 Assistant Ready!** Ask me anything about the event."
    if not vector_store:
        status_msg += "\n\n*Warning: './chroma_db' directory missing. Please run the web crawler first.*"
        
    await cl.Message(content=status_msg).send()

@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    inputs = {"messages": [("user", message.content)]}
    
    final_reply = cl.Message(content="")
    
    async for chunk in agent.astream(inputs, stream_mode="updates"):
        if "agent" in chunk:
            messages = chunk["agent"]["messages"]
            if messages:
                final_reply.content = messages[-1].content
                
    await final_reply.send()