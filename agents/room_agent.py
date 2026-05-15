import json
from typing import Any
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.course import Course
from schemas.timetable import Proposal

class RoomAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def get_instruction(self) -> str:
        return """
            You are the RoomAgent in a timetable scheduling system.
            Your sole job is to assign the most suitable room for a course proposal.

            Rules:
            - Match room type to course requirement (lab course → lab room, non-lab course → classroom)
            - Never assign a room already confirmed at the same timeslot
            - If multiple suitable rooms are free, prefer the one that best fits the course
            - DO NOT MARK policy_approved. Only set the room_id
            
            Call log_decision to explain your reasoning. Then return the proposal with room_id set.
        """

    def get_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
        booked_at_slot = [
            a for a in deps.store.get_assignments()
            if a.timeslot.day == proposal.timeslot.day
            and a.timeslot.start_hour == proposal.timeslot.start_hour
        ]
        return f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}
            Course:
            {json.dumps(course.model_dump(), indent=2)}
            All rooms:
            {json.dumps([r.model_dump() for r in deps.rooms], indent=2)}
            Rooms already confirmed at this timeslot:
            {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}
        """

    async def run(self, proposal: Proposal, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
        prompt = self.get_prompt(proposal, course, deps)
        if failure_reason is not None:
            prompt = f"{prompt}\n\n{self.get_failure_prompt(failure_reason)}"
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())

        return result.output