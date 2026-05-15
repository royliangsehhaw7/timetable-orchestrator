import json
from typing import Any
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.course import Course
from schemas.timetable import Proposal

class PolicyAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def get_instruction(self) -> str:
        return """
            You are the PolicyAgent in a timetable scheduling system.
            Your sole job is to check a fully assembled proposal against all school policy rules.

            Rules:
            - The day must be a valid school day
            - The timeslot must fall entirely within school hours (start and end)
            - The timeslot must not overlap with the lunch break
            - The room type must match the course requirement

            If all rules pass, set policy_approved=True.
            If any rule fails, set policy_approved=False and policy_reason to a specific actionable
            reason identifying exactly which rule was violated.

            Call log_decision to explain your assessment. Then return the proposal with policy_approved and policy_reason set.
        """

    def get_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
        room = next(r for r in deps.rooms if r.id == proposal.room_id)
        # ignore lecturers for now
        # lecturer = next(l for l in deps.lecturers if l.id == proposal.lecturer_id)
        # PROMOT Lecturer:
        #        {json.dumps(lecturer.model_dump(), indent=2)}

        return f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}
            Course:
            {json.dumps(course.model_dump(), indent=2)}
            Room:
            {json.dumps(room.model_dump(), indent=2)}
            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}
        """

    async def run(self, proposal: Proposal, course: Course, deps: Deps) -> Any:
        prompt = self.get_prompt(proposal, course, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())

        return result.output