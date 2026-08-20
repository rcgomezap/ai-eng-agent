from datetime import date

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HotelInfo(StrictModel):
    name: str
    address: str
    city: str
    country: str
    phone_number: str | None = None
    email: str | None = None
    website: str | None = None


class StayDates(StrictModel):
    check_in: date
    check_out: date

    @model_validator(mode="after")
    def validate_date_order(self) -> "StayDates":
        if self.check_out <= self.check_in:
            raise ValueError("check_out must be after check_in")
        return self


class HotelAvailability(StrictModel):
    hotel_id: str
    available_rooms: int = Field(ge=0)
    check_in_date: date
    check_out_date: date


class BookingResult(StrictModel):
    hotel_id: str
    check_in_date: date
    check_out_date: date
    confirmed: bool
