import asyncio
from dataclasses import dataclass, field
from datetime import date

from interfaces import (
    HotelClient,
    HotelNotFoundError,
    HotelUnavailableError,
    InvalidStayDatesError,
)
from schemas import BookingResult, HotelAvailability, HotelInfo


@dataclass(frozen=True, slots=True)
class HotelReservation:
    check_in: date
    check_out: date


@dataclass(slots=True)
class Hotel:
    info: HotelInfo
    reservations: list[HotelReservation] = field(default_factory=lambda: list[HotelReservation]())

    def is_available(self, check_in: date, check_out: date) -> bool:
        return all(
            check_out <= reservation.check_in or check_in >= reservation.check_out
            for reservation in self.reservations
        )


def initialize_hotels() -> list[Hotel]:
    return [
        Hotel(
            HotelInfo(
                name="Grand Plaza",
                address="123 Main St",
                city="Metropolis",
                country="Countryland",
                phone_number="123-456-7890",
                email="info@grandplaza.com",
            )
        ),
        Hotel(
            HotelInfo(
                name="Sunset Resort",
                address="456 Beach Ave",
                city="Coastal City",
                country="Countryland",
                phone_number="098-765-4321",
                email="info@sunsetresort.com",
            )
        ),
        Hotel(
            HotelInfo(
                name="Mountain Retreat",
                address="789 Alpine Rd",
                city="Highland",
                country="Countryland",
                phone_number="555-123-4567",
                email="info@mountainretreat.com",
            )
        ),
    ]


class MockHotelClient(HotelClient):
    def __init__(self) -> None:
        self._hotels = initialize_hotels()
        self._booking_lock = asyncio.Lock()

    @staticmethod
    def _validate_dates(check_in: date, check_out: date) -> None:
        if check_out <= check_in:
            raise InvalidStayDatesError("Checkout must be after check-in.")

    def _find_hotel(self, hotel_id: str) -> Hotel:
        normalized_id = hotel_id.casefold().strip()
        for hotel in self._hotels:
            if hotel.info.name.casefold() == normalized_id:
                return hotel
        raise HotelNotFoundError(f"Hotel '{hotel_id}' was not found.")

    async def get_hotel_info(self, hotel_id: str) -> HotelInfo:
        await asyncio.sleep(0)
        return self._find_hotel(hotel_id).info

    async def search_hotels(self, query: str) -> list[HotelInfo]:
        await asyncio.sleep(0)
        normalized_query = query.casefold().strip()
        if not normalized_query:
            return [hotel.info for hotel in self._hotels]
        return [
            hotel.info
            for hotel in self._hotels
            if normalized_query
            in (
                f"{hotel.info.name} {hotel.info.address} {hotel.info.city} {hotel.info.country}"
            ).casefold()
        ]

    async def book_hotel(self, hotel_id: str, check_in: date, check_out: date) -> BookingResult:
        self._validate_dates(check_in, check_out)
        async with self._booking_lock:
            hotel = self._find_hotel(hotel_id)
            if not hotel.is_available(check_in, check_out):
                raise HotelUnavailableError(
                    f"Hotel '{hotel.info.name}' is unavailable for those dates."
                )
            hotel.reservations.append(HotelReservation(check_in, check_out))
        return BookingResult(
            hotel_id=hotel.info.name,
            check_in_date=check_in,
            check_out_date=check_out,
            confirmed=True,
        )

    async def check_availability(
        self, hotel_id: str, check_in: date, check_out: date
    ) -> HotelAvailability:
        self._validate_dates(check_in, check_out)
        await asyncio.sleep(0)
        hotel = self._find_hotel(hotel_id)
        return HotelAvailability(
            hotel_id=hotel.info.name,
            available_rooms=int(hotel.is_available(check_in, check_out)),
            check_in_date=check_in,
            check_out_date=check_out,
        )

    async def list_hotels(self, check_in: date, check_out: date) -> list[HotelInfo]:
        self._validate_dates(check_in, check_out)
        await asyncio.sleep(0)
        return [hotel.info for hotel in self._hotels if hotel.is_available(check_in, check_out)]
