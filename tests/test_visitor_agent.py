from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage

from ejc_bot.agents.visitor import build_visitor_agent


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model that replays scripted messages and accepts tool binding."""

    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_calls_schedule_tool_and_answers(schedule_db):
    sql = "SELECT title, location FROM events WHERE title LIKE '%Cinema%'"
    model = ScriptedToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "query_event_schedule_sql", "args": {"sql_query": sql}, "id": "1"}
                    ],
                ),
                AIMessage(content="The Open-Air Cinema is at Panorama."),
            ]
        )
    )

    result = build_visitor_agent(model).invoke({"messages": [("user", "Where is the cinema?")]})

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "Panorama" in tool_msg.content
    assert result["messages"][-1].content == "The Open-Air Cinema is at Panorama."


def test_rag_tool_reports_missing_vector_store(schedule_db):
    model = ScriptedToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "database_rag_search", "args": {"query": "tickets"}, "id": "1"}
                    ],
                ),
                AIMessage(content="Sorry, no data yet."),
            ]
        )
    )

    result = build_visitor_agent(model).invoke({"messages": [("user", "Ticket prices?")]})

    tool_msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "ejc-ingest-web" in tool_msg.content
