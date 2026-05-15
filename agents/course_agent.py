import json
from typing import Any
from pydantic_ai import Agent

from .base_agent import BaseAgent
from core.deps import Deps
from schemas.course import Course

class CourseAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    # main agent system_instruction
    def get_instruction(self):
        return f"""
            You are the CourseAgent in a timetable scheduling system. Propose the best timeslot for this course. Return the full proposal with timeslot set.

            Rules:
            - Must follow the allocated hours specified for each course
            - The timeslot must be on a valid school day, within school hours, and not during the lunch break
            - Avoid timeslots already taken by other courses
            - Prefer to spread courses across the week rather than clustering them
            - Consider the course type — lab courses benefit from longer uninterrupted blocks
            - DO NOT MARK policy_approved, leave it

            Call log_decision to explain your reasoning. Then return the proposal with timeslot set.            
        """
    # main agent user instruction
    def get_prompt(self, course: Course, deps: Deps):
        assignments = deps.store.get_assignments()
        
        return f"""
            Course:
            {json.dumps(course.model_dump(), indent=2)}
            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}
            Already confirmed assignments (timeslots already taken):
            {json.dumps([a.model_dump() for a in assignments], indent=2)}

        """

    # run the agent with wrapper agent prompts and instructions
    async def run(self, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
        prompt = self.get_prompt(course, deps)

        if failure_reason is not None:
            prompt = f""" 
                {prompt} 
                {self.get_failure_prompt(failure_reason)}
            """

        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())

        return result.output