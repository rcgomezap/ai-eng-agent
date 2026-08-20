# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import Field

from graph import TOOLS, HotelAgent
from hotel_client import MockHotelClient


class ScriptedChatModel(FakeMessagesListChatModel):
    calls: list[list[BaseMessage]] = Field(default_factory=lambda: list[list[BaseMessage]]())

    def bind_tools(
        self,
        tools: Sequence[object],
        *,
        tool_choice: str | None = None,
        **kwargs: object,
    ) -> Runnable[object, AIMessage]:
        del tools, tool_choice, kwargs
        return cast(Runnable[object, AIMessage], self)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        self.calls.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)


def _model(responses: list[AIMessage]) -> ScriptedChatModel:
    return ScriptedChatModel(responses=list[BaseMessage](responses))


def test_tool_schemas_hide_runtime_dependencies() -> None:
    schemas = {tool.name: set(tool.args) for tool in TOOLS}

    assert schemas["search_hotels"] == {"query"}
    assert "runtime" not in schemas["book_hotel"]
    assert "hotel_client" not in schemas["book_hotel"]


async def test_toolnode_returns_errors_to_model_for_clarification() -> None:
    model = _model(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_hotel_info",
                        "args": {"hotel_id": "Unknown Hotel"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I could not find that hotel. What is its exact name?"),
        ]
    )
    agent = HotelAgent(model, MockHotelClient(), InMemorySaver())

    response = await agent.invoke("Tell me about Unknown Hotel", "error-thread")

    assert "exact name" in str(response.content)
    assert len(model.calls) == 2
    tool_messages = [message for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 1
    assert "not found" in str(tool_messages[0].content)


async def test_same_thread_restores_history() -> None:
    model = _model([AIMessage(content="First answer"), AIMessage(content="Second answer")])
    agent = HotelAgent(model, MockHotelClient(), InMemorySaver())

    await agent.invoke("Remember this", "memory-thread")
    await agent.invoke("What did I say?", "memory-thread")

    second_call_text = " ".join(str(message.content) for message in model.calls[1])
    assert "Remember this" in second_call_text
    assert "First answer" in second_call_text


async def test_sqlite_memory_survives_agent_recreation(tmp_path: Path) -> None:
    database = tmp_path / "memory.sqlite3"
    first_model = _model([AIMessage(content="Stored answer")])
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        first_agent = HotelAgent(first_model, MockHotelClient(), saver)
        await first_agent.invoke("Persist this message", "durable-thread")

    second_model = _model([AIMessage(content="Recovered answer")])
    async with AsyncSqliteSaver.from_conn_string(str(database)) as saver:
        second_agent = HotelAgent(second_model, MockHotelClient(), saver)
        await second_agent.invoke("Recall it", "durable-thread")

    restored_text = " ".join(str(message.content) for message in second_model.calls[0])
    assert "Persist this message" in restored_text
    assert "Stored answer" in restored_text
