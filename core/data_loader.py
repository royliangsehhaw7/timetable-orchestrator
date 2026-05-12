import json
from pathlib import Path


from schemas.course import Course
from schemas.lecturer import Lecturer
from schemas.policy import Policy
from schemas.room import Room


def load_data() -> tuple[list[Course], list[Room], list[Lecturer], Policy]:
    courses   = [Course.model_validate(c)    for c in json.loads(Path("data/courses.json").read_text())]
    rooms     = [Room.model_validate(r)      for r in json.loads(Path("data/rooms.json").read_text())]
    lecturers = [Lecturer.model_validate(l)  for l in json.loads(Path("data/lecturers.json").read_text())]
    policy    = Policy.model_validate(json.loads(Path("data/policy.json").read_text()))
    
    return courses, rooms, lecturers, policy