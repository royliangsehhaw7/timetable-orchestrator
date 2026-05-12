from pydantic import BaseModel

class TimeSlot(BaseModel):
    day: str          # e.g. "Monday"
    start_hour: int   # e.g. 10
    end_hour: int     # e.g. 11