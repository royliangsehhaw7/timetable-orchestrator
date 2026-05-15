import json
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.timetable import Assignment, OrchestratorDecision, Proposal, RejectionRecord

MAX_RETRIES = 5

class OrchestratorAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def get_instruction(self) -> str:
        return f"""
            You are the OrchestratorAgent in a timetable scheduling system.
            You are the sole driver of the scheduling process.
            Every cycle you receive the full system state and decide exactly one next action.

            Decision rules:
            - If no proposal is in-flight and courses remain → pick the best next course, return next_action="dispatch_course" with course_id set
            - If proposal has no timeslot → return next_action="dispatch_course" with failure_context if retrying
            - If proposal has timeslot but no room → return next_action="dispatch_room"
            - If proposal has timeslot, room but policy not yet checked → return next_action="dispatch_policy"
            - If policy_approved is True → return next_action="confirm"
            - If policy_approved is False → read policy_reason, identify the responsible worker, return the correct dispatch action with failure_context explaining what to fix
            - If a course appears {MAX_RETRIES} or more times in the rejection log → return next_action="abandon" with course_id set
            - If no courses remain unscheduled → return next_action="done"

            Call log_decision to explain your reasoning. Then return your decision.
        """

    def get_prompt(self,
        unscheduled: list[str],
        assignments: list[Assignment],
        rejection_log: list[RejectionRecord],
        proposal: Proposal | None,
    ) -> str:
        return f"""
            Current system state:

            Unscheduled courses:
            {json.dumps(unscheduled, indent=2)}
            Confirmed assignments so far:
            {json.dumps([a.model_dump() for a in assignments], indent=2)}
            Rejection log (use this to count retries per course):
            {json.dumps([r.model_dump() for r in rejection_log], indent=2)}
            Current in-flight proposal (null if no proposal is active):
            {json.dumps(proposal.model_dump() if proposal else None, indent=2)}
        """


    async def run(self,
        unscheduled: list[str],
        assignments: list[Assignment],
        rejection_log: list[RejectionRecord],
        proposal: Proposal | None,
        deps: Deps,
    ) -> OrchestratorDecision:
        prompt = self.get_prompt(unscheduled, assignments, rejection_log, proposal)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        return result.output