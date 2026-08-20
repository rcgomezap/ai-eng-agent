import asyncio
from datetime import date

import pytest

from hotel_client import MockHotelClient
from interfaces import HotelUnavailableError, InvalidStayDatesError


async def test_booking_changes_availability() -> None:
    client = MockHotelClient()
    check_in = date(2027, 1, 10)
    check_out = date(2027, 1, 12)

    before = await client.check_availability("Grand Plaza", check_in, check_out)
    booking = await client.book_hotel("Grand Plaza", check_in, check_out)
    after = await client.check_availability("Grand Plaza", check_in, check_out)

    assert before.available_rooms == 1
    assert booking.confirmed is True
    assert after.available_rooms == 0


async def test_invalid_dates_are_rejected() -> None:
    client = MockHotelClient()

    with pytest.raises(InvalidStayDatesError):
        await client.check_availability("Grand Plaza", date(2027, 1, 12), date(2027, 1, 12))


async def test_concurrent_overlapping_bookings_are_atomic() -> None:
    client = MockHotelClient()
    dates = (date(2027, 2, 10), date(2027, 2, 12))

    results = await asyncio.gather(
        client.book_hotel("Sunset Resort", *dates),
        client.book_hotel("Sunset Resort", *dates),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    assert sum(isinstance(result, HotelUnavailableError) for result in results) == 1


async def test_search_matches_city() -> None:
    hotels = await MockHotelClient().search_hotels("highland")

    assert [hotel.name for hotel in hotels] == ["Mountain Retreat"]
