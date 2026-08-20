import os

from langchain.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState # type: ignore
from langgraph.prebuilt import ToolNode
from langchain.tools import tool
from pydantic import BaseModel

from interfaces import HotelClient
from schemas import HotelAvailability, HotelInfo

system_prompt = """
You are a helpful assistant that can provide information about hotels, search for hotels, check availability, and book hotels.

If you are asked to perform an action that requires hotel information, you should use the provided tools to get the necessary data. Always ensure that you provide accurate and up-to-date information based on the user's request.

If you are asked to book a hotel, you should confirm the booking with the user before proceeding. Always provide clear and concise responses, and if you cannot fulfill a request, explain why and suggest alternative actions if possible.

If you are asked to check availability, you should provide the user with the available options and any relevant details.

If you are asked to search for hotels, you should provide a list of hotels that match the user's query, along with relevant details such as location, price, and amenities.

If you are asked to provide hotel information, you should provide the user with the requested details, including location, price, amenities, and any other relevant information.

If you are asked to perform an action that is not related to hotels, you should politely decline and suggest that the user ask a different question or provide more information about their request.

If you are asked to perform an action that requires personal information, you should not ask for or store any personal data. Always prioritize user privacy and data security.
"""

class State(MessagesState):
    active_reservations: list[HotelInfo]

class AppContext(BaseModel):
    hotel_client: HotelClient

@tool
async def get_hotel_info(hotel_id: str, context: AppContext) -> HotelInfo | None:
    """
    Get hotel information by hotel ID.
    """
    return await context.hotel_client.get_hotel_info(hotel_id)

@tool 
async def search_hotels(query: str, context: AppContext) -> list[HotelInfo]:
    """
    Search for hotels based on a query string.
    """
    return await context.hotel_client.search_hotels(query)

@tool
async def book_hotel(hotel_id: str, check_in: str, check_out: str, context: AppContext) -> bool:
    """
    Book a hotel for the specified dates.
    """
    return await context.hotel_client.book_hotel(hotel_id, check_in, check_out)

@tool
async def check_availability(hotel_id: str, check_in: str, check_out: str, context: AppContext) -> HotelAvailability:
    """
    Check the availability of a hotel for the specified dates.
    """
    try:
        availability = await context.hotel_client.check_availability(hotel_id, check_in, check_out)
        return availability
    except ValueError as e:
        raise ValueError(f"Error checking availability: {str(e)}")


tools = [get_hotel_info, search_hotels, book_hotel, check_availability]

llm = ChatOpenAI(model="gpt-4o", temperature=0.7, base_url=os.environ.get("OPENAI_API_BASE_URL"))

llm_with_tools = llm.bind_tools(tools) # pyright: ignore[reportUnknownMemberType]

async def llm_call(state: State):
    system_message = SystemMessage(content=system_prompt)
    messages = [system_message] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"response": response.content} # pyright: ignore[reportUnknownVariableType]

async def 