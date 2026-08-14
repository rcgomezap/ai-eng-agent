import asyncio

from interfaces import HotelClient
import random
from schemas import HotelAvailability, HotelInfo


class Hotel:
    def __init__(self, hotel_info: HotelInfo):
        self.hotel_info = hotel_info

    def add_reservation(self, reservation: HotelReservation) -> bool:
        # Randomly determine if the reservation can be made for simplicity
        if random.random() < 0.1:
            return False
        return True

    def check_availability(self, check_in: str, check_out: str) -> bool:
        # Randomly determine availability for simplicity
        return random.choice([True, False])


def initialize_hotels() -> list[Hotel]:
    hotels = [
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
    return hotels


class HotelReservation:
    def __init__(self, hotel: Hotel, check_in: str, check_out: str):
        self.hotel = hotel
        self.check_in = check_in
        self.check_out = check_out


class MockHotelClient(HotelClient):
    def __init__(self):
        self.hotels = initialize_hotels()

    # Simulate delays in async methods for realism
    async def _delay(self):
        # Random delay between 0.1 and 1 second to simulate network latency
        await asyncio.sleep(random.uniform(0.1, 1.0))

    async def get_hotel_info(self, hotel_id: str) -> HotelInfo | None:
        await self._delay()
        for hotel in self.hotels:
            if hotel.hotel_info.name == hotel_id:
                return hotel.hotel_info
        return None

    async def search_hotels(self, query: str) -> list[HotelInfo]:
        await self._delay()
        return [
            hotel.hotel_info
            for hotel in self.hotels
            if query.lower() in hotel.hotel_info.name.lower()
        ]

    async def book_hotel(self, hotel_id: str, check_in: str, check_out: str) -> bool:
        await self._delay()
        for hotel in self.hotels:
            if hotel.hotel_info.name == hotel_id:
                reservation = HotelReservation(hotel, check_in, check_out)
                return hotel.add_reservation(reservation)
        raise ValueError("Hotel not found")

    async def check_availability(
        self, hotel_id: str, check_in: str, check_out: str
    ) -> HotelAvailability:
        await self._delay()
        for hotel in self.hotels:
            if hotel.hotel_info.name == hotel_id:
                available = hotel.check_availability(check_in, check_out)
                return HotelAvailability(
                    hotel_id=hotel.hotel_info.name,
                    available_rooms=1 if available else 0,
                    check_in_date=check_in,
                    check_out_date=check_out,
                )
        raise ValueError("Hotel not found")

    async def list_hotels(self, check_in: str, check_out: str) -> list[HotelInfo]:
        await self._delay()
        return [
            hotel.hotel_info
            for hotel in self.hotels
            if hotel.check_availability(check_in, check_out)
        ]
