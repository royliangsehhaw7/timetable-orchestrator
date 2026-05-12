from pydantic import BaseModel
from .timeslot import TimeSlot

class Lecturer(BaseModel):
    id: str
    name: str
    courses_taught: list[str]
    unavailable_slots: list[TimeSlot]