import os

import chainlit as cl
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.prebuilt import create_react_agent

# --- 1. SETUP LOCAL VECTOR STORE ---

DB_PATH = "./chroma_db"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

if os.path.exists(DB_PATH):
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
else:
    vector_store = None
    retriever = None

# --- 2. DEFINE SYSTEM AGENT TOOLS ---

@tool
def database_rag_search(query: str) -> str:
    """Search internal database for information about EJA, EJC 2026, schedules, venues, and event details."""
    if not retriever:
        return "Error: Database './chroma_db' folder not found. Please run the crawler script first."
    
    matching_docs = retriever.invoke(query)
    if not matching_docs:
        return "No relevant information found in the event database."
    
    context_blocks = []
    for doc in matching_docs:
        source_url = doc.metadata.get("source", "Unknown source")
        page_title = doc.metadata.get("title", source_url)
        context_blocks.append(
            f"[Source Title: {page_title}]\n[Source URL: {source_url}]\nContent:\n{doc.page_content}"
        )
        
    return "\n\n---\n\n".join(context_blocks)

tools = [database_rag_search]

# --- 3. AGENT SYSTEM PROMPT ---

SYSTEM_PROMPT = """You are a helpful event assistant for EJA and EJC 2026. 
Use the `database_rag_search` tool to look up accurate information about the events.

CRITICAL CITATION RULES:
1. Always base your answers strictly on the retrieved context.
2. Whenever you present facts retrieved from the database, you MUST include markdown source links at the bottom of your response.
3. Format links like this: `[Page Title](URL)`.
4. If multiple sources were used, list them cleanly under a "### Sources" header.
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
    
    status_msg = "🤖 **EJA / EJC 2026 Assistant Ready!** Ask me anything about the event."
    if not vector_store:
        status_msg += "\n\n⚠️ *Warning: './chroma_db' directory missing. Please run the web crawler first.*"
        
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