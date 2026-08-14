from typing import Protocol

from schemas import HotelAvailability, HotelInfo


class HotelClient(Protocol):
    async def get_hotel_info(self, hotel_id: str) -> HotelInfo | None: ...

    async def search_hotels(self, query: str) -> list[HotelInfo]: ...

    async def book_hotel(
        self, hotel_id: str, check_in: str, check_out: str
    ) -> bool: ...

    async def check_availability(
        self, hotel_id: str, check_in: str, check_out: str
    ) -> HotelAvailability: ...

    async def list_hotels(self, check_in: str, check_out: str) -> list[HotelInfo]: ...
