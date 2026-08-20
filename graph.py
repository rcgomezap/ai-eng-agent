# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

from injector import inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, ToolRuntime, tools_condition

from interfaces import HotelClient, HotelError

SYSTEM_PROMPT = """
You are an autonomous hotel assistant. Decide for yourself whether a tool is needed from
the user's request. Never invent hotel data.

When a tool reports an error, inspect it and either make one corrected tool call when the
correction is unambiguous, or ask the user for the missing information. Do not repeat the
same failing call indefinitely.

Before booking, obtain an explicit confirmation from the user for the hotel and exact
dates. Only then call book_hotel with user_confirmed=true. Dates must use ISO format
YYYY-MM-DD. For unrelated requests, explain that you can only assist with hotels.
""".strip()

MAX_CONTEXT_MESSAGES = 40
DEFAULT_RECURSION_LIMIT = 10


class AgentError(Exception):
    """Base exception for failures exposed by the agent service."""


class AgentInputError(AgentError):
    """Raised when an agent invocation has invalid input."""


class ModelInvocationError(AgentError):
    """Raised when the chat model cannot produce a response."""


class AgentLoopLimitError(AgentError):
    """Raised when the graph exceeds its configured reasoning limit."""


class ToolOperationError(Exception):
    """Expected error returned to the model so it can recover or ask a question."""


@dataclass(frozen=True, slots=True)
class AppContext:
    hotel_client: HotelClient


def _serialize_model(model: object) -> dict[str, object]:
    model_dump = getattr(model, "model_dump", None)
    if not callable(model_dump):
        raise TypeError(f"Tool returned an unsupported value: {type(model).__name__}")
    return cast(dict[str, object], model_dump(mode="json"))


@tool
async def get_hotel_info(hotel_id: str, runtime: ToolRuntime[object, object]) -> dict[str, object]:
    """Return verified details for one hotel.

    Use this tool when the user asks for the address, city, country, telephone, email, or
    website of a specific hotel. ``hotel_id`` is currently the complete hotel name, for
    example ``Grand Plaza``. If the name is uncertain, call ``search_hotels`` first rather
    than guessing. A not-found error should lead to a corrected search or a clarification.
    """
    try:
        context = cast(AppContext, runtime.context)
        return _serialize_model(await context.hotel_client.get_hotel_info(hotel_id))
    except HotelError as exc:
        raise ToolOperationError(str(exc)) from exc


@tool
async def search_hotels(
    query: str, runtime: ToolRuntime[object, object]
) -> list[dict[str, object]]:
    """Search the hotel catalog using a name, city, country, or address fragment.

    Use this tool whenever the user describes a destination or only part of a hotel name.
    Matching is case-insensitive. Pass an empty query only when the user explicitly asks to
    list every hotel. An empty result means no catalog entry matched and should be reported
    honestly or followed by a request for a broader query.
    """
    try:
        context = cast(AppContext, runtime.context)
        hotels = await context.hotel_client.search_hotels(query)
        return [_serialize_model(hotel) for hotel in hotels]
    except HotelError as exc:
        raise ToolOperationError(str(exc)) from exc


@tool
async def check_availability(
    hotel_id: str,
    check_in: date,
    check_out: date,
    runtime: ToolRuntime[object, object],
) -> dict[str, object]:
    """Check room availability for a known hotel and exact stay dates.

    Use this before offering or booking a stay. ``hotel_id`` must be the complete hotel
    name. ``check_in`` and ``check_out`` must be ISO dates and checkout must be later than
    check-in. If a date or hotel is missing, ask the user instead of inventing it. If this
    tool returns an error, correct an obvious typo once or ask for clarification.
    """
    try:
        context = cast(AppContext, runtime.context)
        availability = await context.hotel_client.check_availability(hotel_id, check_in, check_out)
        return _serialize_model(availability)
    except HotelError as exc:
        raise ToolOperationError(str(exc)) from exc


@tool
async def book_hotel(
    hotel_id: str,
    check_in: date,
    check_out: date,
    user_confirmed: bool,
    runtime: ToolRuntime[object, object],
) -> dict[str, object]:
    """Create a hotel reservation after explicit user confirmation.

    Call this tool only after availability has been checked and the user has explicitly
    confirmed the complete hotel name plus exact check-in and checkout dates. Set
    ``user_confirmed`` to true only when that confirmation exists in the conversation.
    Never retry this mutating operation automatically after an unexpected service or
    network failure because doing so could create duplicate bookings.
    """
    if not user_confirmed:
        raise ToolOperationError("Explicit user confirmation is required before booking.")
    try:
        context = cast(AppContext, runtime.context)
        result = await context.hotel_client.book_hotel(hotel_id, check_in, check_out)
        return _serialize_model(result)
    except HotelError as exc:
        raise ToolOperationError(str(exc)) from exc


TOOLS: Sequence[BaseTool] = (
    get_hotel_info,
    search_hotels,
    check_availability,
    book_hotel,
)


class HotelAgent:
    def __init__(
        self,
        model: BaseChatModel,
        hotel_client: HotelClient,
        checkpointer: BaseCheckpointSaver[str],
    ) -> None:
        self._context = AppContext(hotel_client=hotel_client)
        model_with_tools = model.bind_tools(TOOLS)

        async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            history = list(state["messages"][-MAX_CONTEXT_MESSAGES:])
            while history and not isinstance(history[0], HumanMessage):
                history.pop(0)
            try:
                response = await model_with_tools.ainvoke(
                    [SystemMessage(content=SYSTEM_PROMPT), *history]
                )
            except Exception as exc:
                raise ModelInvocationError("The language model request failed.") from exc
            return {"messages": [response]}

        builder = StateGraph(MessagesState, context_schema=AppContext)
        builder.add_node("agent", call_model)
        builder.add_node(
            "tools",
            ToolNode(TOOLS, handle_tool_errors=(ToolOperationError,)),
        )
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        self._graph: CompiledStateGraph[MessagesState, AppContext, MessagesState, MessagesState] = (
            builder.compile(checkpointer=checkpointer)
        )

    async def invoke(self, prompt: str, thread_id: str) -> AIMessage:
        clean_prompt = prompt.strip()
        clean_thread_id = thread_id.strip()
        if not clean_prompt:
            raise AgentInputError("The prompt cannot be empty.")
        if not clean_thread_id:
            raise AgentInputError("The thread_id cannot be empty.")

        try:
            result = await self._graph.ainvoke(
                {"messages": [HumanMessage(content=clean_prompt)]},
                config={
                    "configurable": {"thread_id": clean_thread_id},
                    "recursion_limit": DEFAULT_RECURSION_LIMIT,
                },
                context=self._context,
            )
        except GraphRecursionError as exc:
            raise AgentLoopLimitError(
                f"The agent exceeded the {DEFAULT_RECURSION_LIMIT}-step reasoning limit."
            ) from exc

        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                return message
        raise ModelInvocationError("The graph completed without an assistant response.")


class AgentFactory:
    @inject
    def __init__(self, model: BaseChatModel, hotel_client: HotelClient) -> None:
        self._model = model
        self._hotel_client = hotel_client

    def create(self, checkpointer: BaseCheckpointSaver[str]) -> HotelAgent:
        return HotelAgent(self._model, self._hotel_client, checkpointer)
