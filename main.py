import asyncio
import logging

from pydantic_ai import Agent
from tools.agent_logger import log_decision

from agents.orchestrator_agent import OrchestratorAgent
from agents.course_agent import CourseAgent
from agents.room_agent import RoomAgent
from agents.policy_agent import PolicyAgent

from core.data_loader import load_data
from core.deps import Deps
from stores.store import Store
from schemas.timetable import OrchestratorDecision, Proposal
from core.llm_factory import LLMFactory
from control.coordinator import Coordinator

async def main():
    #  --- load
    courses, rooms, lecturers, policy = load_data()
    #  --- prepare store
    store = Store()
    store.seed([c.id for c in courses])
    #
    deps = Deps(
        store = store,
        courses = courses,
        rooms = rooms,
        lecturers = lecturers,
        policy = policy,
        total_tokens = 0
    )
    # --- create all agents
    factory = LLMFactory("openrouter")
    model = factory.get_model(model="nvidia/nemotron-3-super-120b-a12b:free")

    oAgent = Agent(
        model = model,
        output_type=OrchestratorDecision,
        tools = [log_decision]
    )
    orchestratorAgent = OrchestratorAgent(name="OrchestratorAgent",agent=oAgent)
    cAgent = Agent(
        model=model,
        output_type=Proposal,
        tools = [log_decision]
    )
    courseAgent = CourseAgent(name="CourseAgent", agent=cAgent)
    rAgent = Agent(
        model = model,
        output_type = Proposal,
        tools=[log_decision]
    )
    roomAgent = RoomAgent(name="RoomAgent", agent=rAgent)
    pAgent = Agent(
        model = model,
        output_type=Proposal,
        tools=[log_decision]
    )
    policyAgent = PolicyAgent(name="PolicyAgent", agent=pAgent)

    #
    coordinator = Coordinator({"orchestrator": orchestratorAgent, "course": courseAgent, "room": roomAgent, "policy": policyAgent})
    await coordinator.run(deps)

if __name__ == "__main__":
    asyncio.run(main())
