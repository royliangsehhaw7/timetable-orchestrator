from pydantic import BaseModel

class Course(BaseModel):
    id: str
    name: str
    requires_lab: bool
    hours: int