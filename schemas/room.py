from pydantic import BaseModel

class Room(BaseModel):
    id: str
    name: str
    room_type: str    # "lab" or "classroom"