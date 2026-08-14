from typing import Optional

from pydantic import BaseModel


class HotelInfo(BaseModel):
    name: str
    address: str
    city: str
    country: str
    phone_number: Optional[str] = None
    email: Optional[str] = None
    website: Optional[str] = None


class HotelAvailability(BaseModel):
    hotel_id: str
    available_rooms: int
    check_in_date: str
    check_out_date: str
