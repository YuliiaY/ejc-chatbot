"""The visitor-facing assistant: a tool-calling agent over the website RAG and the schedule DB."""

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from ejc_bot.agents.tools import VISITOR_TOOLS
from ejc_bot.config import Settings, get_settings
from ejc_bot.llm import get_chat_model

SYSTEM_PROMPT = """You are a helpful event assistant for the European Juggling Association (EJA) \
and the European Juggling Convention (EJC) {year}.

Tool Usage Guide:
1. For event timing, dates, locations, or schedule lookups, USE `query_event_schedule_sql`.
2. For general questions (accommodations, registration, venue rules), USE `database_rag_search`.

CRITICAL CITATION RULES:
1. Always base your answers strictly on the retrieved context or SQL results.
2. Whenever you present facts retrieved from the database, you MUST include markdown source links \
at the bottom of your response.
3. If you use `query_event_schedule_sql`, you MUST include the source URL of the schedule page \
({program_url}) in your response.
4. Format links like this: `[Page Title](URL)`.
5. If multiple sources were used, list them cleanly under a "### Sources" header.
"""


def build_system_prompt(settings: Settings) -> str:
    return SYSTEM_PROMPT.format(year=settings.convention_year, program_url=settings.program_url)


def build_visitor_agent(model: BaseChatModel | None = None):
    settings = get_settings()
    return create_agent(
        model or get_chat_model(settings),
        VISITOR_TOOLS,
        system_prompt=build_system_prompt(settings),
    )
