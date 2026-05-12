from dataclasses import dataclass
from stores.store import Store
from schemas.course import Course
from schemas.room import Room
from schemas.lecturer import Lecturer
from schemas.policy import Policy

@dataclass
class Deps:
    store: Store                 # mutable — orchestrator writes to this

    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    
    total_tokens: int = 0        # accumulated token usage across all agents