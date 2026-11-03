"""Chainlit chat UI. Run with: ``uv run chainlit run src/ejc_bot/ui/chainlit_app.py``."""

import chainlit as cl

from ejc_bot.agents.visitor import build_visitor_agent
from ejc_bot.config import get_settings
from ejc_bot.stores.vector import get_vector_store


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("agent", build_visitor_agent())

    year = get_settings().convention_year
    status_msg = f"🤹 **EJA / EJC {year} Assistant Ready!** Ask me anything about the event."
    if get_vector_store() is None:
        status_msg += "\n\n*Warning: vector store missing. Please run `ejc-ingest-web` first.*"

    await cl.Message(content=status_msg).send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    result = await agent.ainvoke({"messages": [("user", message.content)]})
    await cl.Message(content=result["messages"][-1].content).send()
