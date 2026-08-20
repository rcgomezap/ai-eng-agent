from datetime import date
from typing import Protocol

from schemas import BookingResult, HotelAvailability, HotelInfo


class HotelError(Exception):
    """Base exception for expected hotel-domain failures."""


class HotelNotFoundError(HotelError):
    """Raised when a requested hotel does not exist."""


class InvalidStayDatesError(HotelError):
    """Raised when checkout is not later than check-in."""


class HotelUnavailableError(HotelError):
    """Raised when a hotel has no inventory for the requested dates."""


class HotelClient(Protocol):
    async def get_hotel_info(self, hotel_id: str) -> HotelInfo: ...

    async def search_hotels(self, query: str) -> list[HotelInfo]: ...

    async def book_hotel(self, hotel_id: str, check_in: date, check_out: date) -> BookingResult: ...

    async def check_availability(
        self, hotel_id: str, check_in: date, check_out: date
    ) -> HotelAvailability: ...

    async def list_hotels(self, check_in: date, check_out: date) -> list[HotelInfo]: ...
