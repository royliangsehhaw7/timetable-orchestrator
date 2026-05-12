from pydantic import BaseModel
from .timeslot import TimeSlot

class Proposal(BaseModel):
    id: str
    course_id: str
    timeslot: TimeSlot | None = None
    room_id: str | None = None
    lecturer_id: str | None = None
    policy_approved: bool | None = False
    policy_reason: str | None = None
    retry_count: int = 0

class Assignment(BaseModel):
    course_id: str
    room_id: str
    lecturer_id: str
    timeslot: TimeSlot

class RejectionRecord(BaseModel):
    course_id: str
    reason: str
    cycle: int

class OrchestratorDecision(BaseModel):
    next_action: str                    # "dispatch_course" | "dispatch_room" | "dispatch_lecturer"
                                        # | "dispatch_policy" | "confirm" | "abandon" | "done"
    course_id: str | None = None        # required for dispatch_course and abandon
    failure_context: str | None = None  # passed to worker on retry
    reason: str                         # orchestrator's explanation of its decision