from pydantic import BaseModel

class Policy(BaseModel):
    school_days: list[str]
    school_start_hour: int
    school_end_hour: int
    lunch_start_hour: int
    lunch_end_hour: int