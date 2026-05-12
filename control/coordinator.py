import uuid
from core.deps import Deps
from core.logger import logger
from schemas.timetable import Proposal, OrchestratorDecision

MAX_CYCLES = 50
MAX_RETRIES = 5

class Coordinator():
    def __init__(self, agents: dict):
        self._agents = agents

    async def run(self, deps: Deps) -> None:
        cycle = 0
        in_flight_proposal: Proposal | None = None

        while cycle < MAX_CYCLES:
            cycle += 1
            logger.info(f"[cycle {cycle}] unscheduled: {deps.store.get_unscheduled_courses()}")

            decision = await self._agents["orchestrator"].run(
                unscheduled   = deps.store.get_unscheduled_courses(),
                assignments   = deps.store.get_assignments(),
                rejection_log = deps.store.get_rejection_log(),
                proposal      = in_flight_proposal,
                deps          = deps,
            )

            logger.info(f"[cycle {cycle}] orchestrator → {decision.next_action}: {decision.reason}")

            in_flight_proposal = await self._execute(decision, in_flight_proposal, self._agents, deps, cycle)

            if decision.next_action == "done":
                break

        # return _produce_output(deps, cycle)

    async def _execute(self,
        decision: OrchestratorDecision,
        proposal: Proposal | None,
        agents: dict,
        deps: Deps,
        cycle: int,
    ) -> Proposal | None:
        course = (
            next((c for c in deps.courses if c.id == decision.course_id), None)
            if decision.course_id else
            next((c for c in deps.courses if c.id == proposal.course_id), None) if proposal else None
        )

        match decision.next_action:
            case "dispatch_course":
                proposal = Proposal(id=str(uuid.uuid4()), course_id=decision.course_id)
                return await agents["course"].run(course, deps, decision.failure_context)

            case "dispatch_room":
                return await agents["room"].run(proposal, course, deps, decision.failure_context)

            case "dispatch_lecturer":
                return await agents["lecturer"].run(proposal, course, deps, decision.failure_context)

            case "dispatch_policy":
                return await agents["policy"].run(proposal, course, deps)

            case "confirm":
                deps.store.confirm(proposal, cycle)
                logger.info(f"[cycle {cycle}] confirmed {proposal.course_id}")
                return None

            case "abandon":
                deps.store.record_rejection(decision.course_id, decision.reason, cycle)
                deps.store.abandon(decision.course_id)
                logger.warning(f"[cycle {cycle}] abandoned {decision.course_id}: {decision.reason}")
                return None

            case "done":
                return None