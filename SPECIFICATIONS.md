# Multi-Agent Timetable Generation System
## Specification v8.0

---

## 1. Purpose

A learning-oriented multi-agent system that generates a valid weekly school timetable. The primary goal is to understand and implement the **Orchestrator-Worker Pattern** in a grounded, practical context using pydantic-ai's deps, agents, and tools features properly.

The timetable domain is intentionally simple. It exists to make data flow visible and traceable, not to solve a hard scheduling problem.

---

## 2. Architecture

### 2.1 Pattern

This system implements the **Orchestrator-Worker Pattern**.

One orchestrator agent is the sole driver of the entire scheduling process. Every cycle, the orchestrator LLM inspects the full system state — what is unscheduled, what has been confirmed, what has been rejected, and what proposal is currently in-flight — and decides what to do next. Python executes that decision, then hands control back to the orchestrator.

Worker agents are domain specialists. They do deep reasoning within their area and return structured results. They never decide what happens next. They never call each other. Only the orchestrator dispatches them.

### 2.2 Agents

There are five pydantic-ai agents:

| Agent | Role |
|-------|------|
| `OrchestratorAgent` | Strategic driver — reads full state, decides next action every cycle |
| `CourseAgent` | Timeslot domain specialist |
| `RoomAgent` | Room fit specialist |
| `LecturerAgent` | Lecturer suitability specialist |
| `PolicyAgent` | Compliance specialist |

All five share a common `BaseAgent`. Workers are LLM-powered agents that reason deeply within their domain.

### 2.3 Flow

Every cycle follows the same structure: the orchestrator sees everything and issues one decision; Python executes it.

```
LOOP:
    OrchestratorAgent
        receives: unscheduled courses, confirmed assignments,
                  rejection log, current in-flight proposal
        returns:  OrchestratorDecision(next_action, course_id,
                                       failure_context, reason)

    execute(decision):
        "dispatch_course"   → CourseAgent.run(course, deps, failure_context)
                              → proposal.timeslot set
        "dispatch_room"     → RoomAgent.run(proposal, course, deps, failure_context)
                              → proposal.room_id set
        "dispatch_lecturer" → LecturerAgent.run(proposal, course, deps, failure_context)
                              → proposal.lecturer_id set
        "dispatch_policy"   → PolicyAgent.run(proposal, course, deps)
                              → proposal.policy_approved + policy_reason set
        "confirm"           → store.confirm(proposal); in_flight = None
        "abandon"           → store.abandon(course_id); in_flight = None
        "done"              → exit loop
```

**What the orchestrator reasons about each cycle:**

- Nothing in flight, courses remain → pick the best next course, dispatch_course
- Proposal has timeslot, no room → dispatch_room
- Proposal has room, no lecturer → dispatch_lecturer
- Proposal fully assembled → dispatch_policy
- Policy approved → confirm
- Policy rejected → read the reason, identify the responsible worker, dispatch with failure_context
- Rejection count for a course hits MAX_RETRIES → abandon
- No courses remain unscheduled → done

### 2.4 What the Orchestrator does not do

The orchestrator does not perform domain reasoning. It does not choose timeslots, rooms, or lecturers. It does not evaluate policy rules. It reads state, decides the next action, and names which worker to dispatch. Domain judgment belongs entirely to the specialist workers.

---

## 3. Project Structure

```
timetable/
├── agents/
│   ├── base_agent.py
│   ├── orchestrator_agent.py
│   ├── course_agent.py
│   ├── room_agent.py
│   ├── lecturer_agent.py
│   └── policy_agent.py
├── control/
│   └── coordinator.py
├── core/
│   ├── data_loader.py
│   ├── deps.py
│   ├── logger.py
│   └── llm_factory.py
├── schemas/
│   ├── timeslot.py
│   ├── course.py
│   ├── room.py
│   ├── lecturer.py
│   ├── policy.py
│   └── timetable.py
├── store/
│   └── store.py
├── tools/
│   └── agent_logger.py
├── data/
│   ├── courses.json
│   ├── rooms.json
│   ├── lecturers.json
│   └── policy.json
└── main.py
```

---

## 4. Schemas

**Location**: `schemas/`

All files are plain Pydantic `BaseModel` classes. No methods, no logic, no imports from any other project module.

**`schemas/timeslot.py`**

```python
class TimeSlot(BaseModel):
    day: str          # e.g. "Monday"
    start_hour: int   # e.g. 10
    end_hour: int     # e.g. 11
```

**`schemas/course.py`**

```python
class Course(BaseModel):
    id: str
    name: str
    requires_lab: bool
```

**`schemas/room.py`**

```python
class Room(BaseModel):
    id: str
    name: str
    room_type: str    # "lab" or "classroom"
```

**`schemas/lecturer.py`**

```python
class Lecturer(BaseModel):
    id: str
    name: str
    courses_taught: list[str]
    unavailable_slots: list[TimeSlot]
```

**`schemas/policy.py`**

```python
class Policy(BaseModel):
    school_days: list[str]
    school_start_hour: int
    school_end_hour: int
    lunch_start_hour: int
    lunch_end_hour: int
```

**`schemas/timetable.py`**

```python
class Proposal(BaseModel):
    id: str
    course_id: str
    timeslot: TimeSlot | None = None
    room_id: str | None = None
    lecturer_id: str | None = None
    policy_approved: bool | None = None
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
```

A `Proposal` is in-flight while it lives in the control loop as a local variable. Once the orchestrator decides `confirm` or `abandon`, Python writes to the store and sets `in_flight_proposal = None`. The store never holds in-flight proposals.

**Verification**: import each schema in a scratch script and instantiate one instance per model with dummy data. All fields should round-trip through `model_dump()` and `model_validate()` without error.

---

## 5. The Store

**Location**: `store/store.py`

The store is a persistence and audit layer. It records confirmed assignments and rejection history, and provides the state snapshot the orchestrator reads every cycle to make decisions. It is not a coordination medium between agents. In-flight proposals live in the control loop as a local variable, not here.

### 5.1 What it holds

```python
class Store:
    def __init__(self):
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []
```

`_unscheduled_courses` is seeded at startup. A course id is removed only when confirmed or abandoned.

### 5.2 Methods

#### Seeding

| Method | Signature | What it does |
|--------|-----------|--------------|
| `seed` | `(course_ids: list[str]) -> None` | Populates `_unscheduled_courses` at startup. Called once before the control loop begins. |

#### Write methods

| Method | Signature | What it does |
|--------|-----------|--------------|
| `confirm` | `(proposal: Proposal, cycle: int) -> None` | Constructs an `Assignment` from the proposal, appends to `_assignments`, removes `course_id` from `_unscheduled_courses`. |
| `record_rejection` | `(course_id: str, reason: str, cycle: int) -> None` | Appends a `RejectionRecord`. Course id remains in `_unscheduled_courses`. |
| `abandon` | `(course_id: str) -> None` | Removes `course_id` from `_unscheduled_courses`. Called by the control loop when the orchestrator returns `next_action="abandon"`. |

#### Read methods

| Method | Signature | What it returns |
|--------|-----------|-----------------|
| `get_assignments` | `() -> list[Assignment]` | All confirmed assignments. |
| `get_rejection_log` | `() -> list[RejectionRecord]` | Full rejection history. Orchestrator uses this to count retries per course. |
| `get_unscheduled_courses` | `() -> list[str]` | Course ids not yet confirmed or abandoned. |

No domain logic lives in the store. No filtering by agent. No validation. Just reads and writes.

**Verification**: seed with three course ids, confirm one, reject one, abandon one. Print all three collections after each operation and confirm state is correct.

---

## 6. Dependencies — Deps and RunContext

### 6.1 What Deps holds

`Deps` is a single container passed into every agent activation. It holds the store (shared mutable state) and the reference data (read-only).

```python
@dataclass
class Deps:
    store: Store                 # mutable — only the control loop writes to this
    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    total_tokens: int = 0        # accumulated token usage across all agents
```

Workers do not write to the store. They receive context in their prompt and return structured results via `output_type`. Only the control loop writes to the store, acting on the orchestrator's decisions.

### 6.2 How Deps flows at runtime

One `Deps` instance is created before the control loop and passed into every agent's `run()` call. pydantic-ai injects it into `RunContext` so the `log_decision` tool can accumulate token usage into `deps.total_tokens`.

---

## 7. Tools

### 7.1 Design decision

There is one tool in this system: `log_decision`. It is registered on every agent including the orchestrator.

A tool is something the LLM calls mid-reasoning to perform an action with a side effect. `log_decision` qualifies: the LLM triggers it deliberately to narrate its reasoning, and it produces structured log output and accumulates token counts as side effects.

Store writes (confirm, record_rejection, abandon) happen in Python inside the control loop after the LLM returns its structured result. They are not tools because the LLM does not trigger them mid-reasoning — Python acts on the returned `output_type`.

### 7.2 The tool

**Location**: `tools/agent_logger.py`

```python
import logging
from pydantic_ai import RunContext
from core.deps import Deps

logger = logging.getLogger(__name__)

async def log_decision(ctx: RunContext[Deps], message: str) -> str:
    """Call this once to explain your reasoning before returning your result.
    Describe what you found, what you decided, and why."""
    usage = ctx.usage
    ctx.deps.total_tokens += usage.total_tokens or 0
    logger.info(
        f"[decision] {message} | "
        f"tokens: request={usage.request_tokens} "
        f"response={usage.response_tokens} "
        f"total={usage.total_tokens}"
    )
    return "logged"
```

### 7.3 Tool registration per agent

`log_decision` is the only tool. It is registered on all five agents.

| Tool | Orchestrator | Course | Room | Lecturer | Policy |
|------|:------------:|:------:|:----:|:--------:|:------:|
| `log_decision` | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 8. Agents

### 8.1 Design principles

Each agent has one responsibility. Each agent is powered by an LLM. The LLM makes the domain decision — which timeslot to propose, which room fits, which lecturer is available, whether policy is satisfied, what action to take next. Python does not make these decisions.

Each agent has two prompt methods:

- `get_instruction()` — returns the agent's stable identity and rules. This is passed to `agent.run()` as the `instructions=` argument, which pydantic-ai sends as the system turn.
- `get_prompt()` — returns the live situational data for this specific call. This is passed as the user turn.

This split matters because the LLM treats the system turn as grounding — who it is and what it must never do — and the user turn as the task to act on right now.

Workers return `Proposal` with the relevant fields filled in. The orchestrator returns `OrchestratorDecision`.

Workers do not read from the store directly. They do not call each other.

### 8.2 BaseAgent

**Location**: `agents/base_agent.py`

```python
from abc import ABC, abstractmethod
from pydantic_ai import Agent

class BaseAgent(ABC):
    def __init__(self, name: str, agent: Agent):
        self._name = name
        self._agent = agent

    @abstractmethod
    def get_instruction(self) -> str: ...

    def get_failure_prompt(self, failure_reason: str) -> str:
        return f"""
            CORRECTION REQUIRED
            Your previous result was rejected for the following reason:
            {failure_reason}

            Return a different result that avoids this problem.
        """
```

**Key points:**

- `get_instruction()` is abstract — every subclass must implement it. It holds the agent's identity and rules. It takes no arguments because it never changes between calls.
- `get_prompt()` is **not** declared on the base. Each subclass defines it with its own typed arguments, since each agent needs different data. There is no meaningful shared signature to enforce here.
- `get_failure_prompt()` is a concrete shared method — not abstract. Any subclass can override it if needed, but the default implementation is used by all workers as-is. Python will use the subclass version if one exists, otherwise falls back to this one.
- `run()` is **not** declared on the base for the same reason as `get_prompt()` — each agent's `run()` takes different arguments.

---

### 8.3 CourseAgent

- **Location**: `agents/course_agent.py`
- **Responsibility**: Given a course and scheduling context, propose the best timeslot.
- **Output type**: `Proposal`

```python
import json
from typing import Any
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.course import Course

class CourseAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def get_instruction(self) -> str:
        return """
            You are the CourseAgent in a timetable scheduling system.
            Your sole job is to propose the best timeslot for a given course.

            Rules:
            - The timeslot must be on a valid school day, within school hours, and not during the lunch break
            - Avoid timeslots already taken by other courses
            - Prefer to spread courses across the week rather than clustering them
            - Consider the course type — lab courses benefit from longer uninterrupted blocks

            Call log_decision to explain your reasoning. Then return the proposal with timeslot set.
        """

    def get_prompt(self, course: Course, deps: Deps) -> str:
        assignments = deps.store.get_assignments()
        return f"""
            Course:
            {json.dumps(course.model_dump(), indent=2)}

            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}

            Already confirmed assignments (timeslots already taken):
            {json.dumps([a.model_dump() for a in assignments], indent=2)}
        """

    async def run(self, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
        prompt = self.get_prompt(course, deps)
        if failure_reason is not None:
            prompt = f"{prompt}\n\n{self.get_failure_prompt(failure_reason)}"
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        return result.data
```

---

### 8.4 RoomAgent

- **Location**: `agents/room_agent.py`
- **Responsibility**: Given a proposal with a timeslot, assign the most suitable room.
- **Output type**: `Proposal`

```python
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
        return result.data
```

---

### 8.5 LecturerAgent

- **Location**: `agents/lecturer_agent.py`
- **Responsibility**: Given a proposal with a timeslot and room, assign the most suitable lecturer.
- **Output type**: `Proposal`

```python
import json
from typing import Any
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.course import Course
from schemas.timetable import Proposal

class LecturerAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def get_instruction(self) -> str:
        return """
            You are the LecturerAgent in a timetable scheduling system.
            Your sole job is to assign the most suitable lecturer for a course proposal.

            Rules:
            - The lecturer must teach this course (check courses_taught)
            - The lecturer must not already be confirmed at this timeslot
            - The lecturer must not have this timeslot in their unavailable_slots
            - If multiple qualified lecturers are free, prefer the one with the lighter confirmed workload so far

            Call log_decision to explain your reasoning. Then return the proposal with lecturer_id set.
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

            All lecturers (includes courses_taught and unavailable_slots):
            {json.dumps([l.model_dump() for l in deps.lecturers], indent=2)}

            Lecturers already confirmed at this timeslot:
            {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}
        """

    async def run(self, proposal: Proposal, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
        prompt = self.get_prompt(proposal, course, deps)
        if failure_reason is not None:
            prompt = f"{prompt}\n\n{self.get_failure_prompt(failure_reason)}"
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        return result.data
```

---

### 8.6 PolicyAgent

- **Location**: `agents/policy_agent.py`
- **Responsibility**: Given a fully assembled proposal, perform a deep compliance check against school policy.
- **Output type**: `Proposal`

```python
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
            - The lecturer must be qualified to teach this course
            - The lecturer must not be unavailable at this timeslot

            If all rules pass, set policy_approved=True.
            If any rule fails, set policy_approved=False and policy_reason to a specific actionable
            reason identifying exactly which rule was violated.

            Call log_decision to explain your assessment. Then return the proposal with policy_approved and policy_reason set.
        """

    def get_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
        room = next(r for r in deps.rooms if r.id == proposal.room_id)
        lecturer = next(l for l in deps.lecturers if l.id == proposal.lecturer_id)
        return f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}

            Course:
            {json.dumps(course.model_dump(), indent=2)}

            Room:
            {json.dumps(room.model_dump(), indent=2)}

            Lecturer:
            {json.dumps(lecturer.model_dump(), indent=2)}

            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}
        """

    async def run(self, proposal: Proposal, course: Course, deps: Deps) -> Any:
        prompt = self.get_prompt(proposal, course, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        return result.data
```

`PolicyAgent` has no `failure_reason` parameter — it is always called with a complete proposal and always returns a verdict. The orchestrator reads that verdict and decides what to do next.

---

### 8.7 OrchestratorAgent

- **Location**: `agents/orchestrator_agent.py`
- **Responsibility**: Read full system state every cycle and decide the next action.
- **Output type**: `OrchestratorDecision`

This is the driver. It is called at the top of every cycle — not just on failure. It sees everything: what is unscheduled, what is confirmed, what has been rejected, and what proposal is currently in-flight. It returns one decision. Python executes it.

```python
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
            - If proposal has timeslot and room but no lecturer → return next_action="dispatch_lecturer"
            - If proposal has timeslot, room, and lecturer but policy not yet checked → return next_action="dispatch_policy"
            - If policy_approved is True → return next_action="confirm"
            - If policy_approved is False → read policy_reason, identify the responsible worker, return the correct dispatch action with failure_context explaining what to fix
            - If a course appears {MAX_RETRIES} or more times in the rejection log → return next_action="abandon" with course_id set
            - If no courses remain unscheduled → return next_action="done"

            Call log_decision to explain your reasoning. Then return your decision.
        """

    def get_prompt(
        self,
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

    async def run(
        self,
        unscheduled: list[str],
        assignments: list[Assignment],
        rejection_log: list[RejectionRecord],
        proposal: Proposal | None,
        deps: Deps,
    ) -> OrchestratorDecision:
        prompt = self.get_prompt(unscheduled, assignments, rejection_log, proposal)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        return result.data
```

---

## 9. Control Flow

**Location**: `control/coordinator.py`

### 9.1 Startup sequence

```
1. Load all JSON data from data/
2. Instantiate Store
3. Call store.seed([c.id for c in courses])
4. Instantiate Deps(store, courses, rooms, lecturers, policy, total_tokens=0)
5. Instantiate all five agents
6. Begin control loop
```

### 9.2 Control loop

The loop is a thin executor. The orchestrator LLM drives every decision. Python reads the decision and calls the right worker or store method.

```python
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
```

**Note on `dispatch_course`**: `proposal` is assigned a fresh `Proposal` object before being passed into `CourseAgent.run()`. The agent receives the course and fills in the timeslot, returning an updated `Proposal`. This replaces the previous bug where `new_proposal` was created but never used.

### 9.3 Termination

| Condition | Outcome |
|-----------|---------|
| Orchestrator returns `next_action="done"` | Success — full timetable produced |
| Orchestrator returns `next_action="abandon"` for a course | That course abandoned, orchestrator picks next |
| `cycle > MAX_CYCLES` | Partial result — report what was scheduled and what remains |

### 9.4 Output

```python
def _produce_output(deps: Deps, cycle: int) -> dict:
    # joins assignments with course, room, lecturer names from reference data
    # joins unscheduled with rejection log for reason reporting
```

---

## 10. Core Infrastructure

### 10.1 Logger

**Location**: `core/logger.py`

```python
import logging

logger = logging.getLogger("timetable")
```

One named logger. All modules that need logging import this and call `logger.info(...)` or `logger.warning(...)` directly at the call site. No wrapper functions.

The `log_decision` tool in `tools/agent_logger.py` uses its own `logging.getLogger(__name__)` as before.

### 10.2 LLM Factory

**Location**: `core/llm_factory.py`

```python
def make_model(provider: str) -> KnownModelName | Model:
    ...
```

Accepts a string like `"openai:gpt-4o"` or `"anthropic:claude-3-5-haiku-latest"` and returns a configured pydantic-ai model object.

### 10.3 Data Loader

**Location**: `core/data_loader.py`

```python
def load_data() -> tuple[list[Course], list[Room], list[Lecturer], Policy]:
    courses   = [Course.model_validate(c)    for c in json.loads(Path("data/courses.json").read_text())]
    rooms     = [Room.model_validate(r)      for r in json.loads(Path("data/rooms.json").read_text())]
    lecturers = [Lecturer.model_validate(l)  for l in json.loads(Path("data/lecturers.json").read_text())]
    policy    = Policy.model_validate(json.loads(Path("data/policy.json").read_text()))
    return courses, rooms, lecturers, policy
```

---

## 11. Data Files

### courses.json
```json
[
  { "id": "CS201", "name": "Data Structures",             "requires_lab": true  },
  { "id": "MA101", "name": "Calculus",                    "requires_lab": false },
  { "id": "CS101", "name": "Introduction to Programming", "requires_lab": true  },
  { "id": "PH201", "name": "Physics I",                   "requires_lab": false }
]
```

### rooms.json
```json
[
  { "id": "R001", "name": "Lecture Theatre 1", "room_type": "classroom" },
  { "id": "R002", "name": "Lecture Theatre 2", "room_type": "classroom" },
  { "id": "R003", "name": "Computer Lab A",    "room_type": "lab"       },
  { "id": "R004", "name": "Computer Lab B",    "room_type": "lab"       }
]
```

### lecturers.json
```json
[
  {
    "id": "L001",
    "name": "Dr. Okafor",
    "courses_taught": ["CS201", "CS101"],
    "unavailable_slots": [
      { "day": "Monday", "start_hour": 8, "end_hour": 10 }
    ]
  },
  {
    "id": "L002",
    "name": "Prof. Singh",
    "courses_taught": ["MA101"],
    "unavailable_slots": []
  },
  {
    "id": "L003",
    "name": "Dr. Reyes",
    "courses_taught": ["PH201"],
    "unavailable_slots": [
      { "day": "Thursday", "start_hour": 14, "end_hour": 16 }
    ]
  }
]
```

### policy.json
```json
{
  "school_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "school_start_hour": 8,
  "school_end_hour": 17,
  "lunch_start_hour": 12,
  "lunch_end_hour": 13
}
```

---

## 12. Output Format

### Success
```json
{
  "generated": true,
  "total_cycles": 14,
  "total_tokens": 18420,
  "assignments": [
    {
      "course_id": "CS201",
      "course_name": "Data Structures",
      "room_id": "R003",
      "room_name": "Computer Lab A",
      "lecturer_id": "L001",
      "lecturer_name": "Dr. Okafor",
      "day": "Wednesday",
      "start_hour": 10,
      "end_hour": 11
    }
  ],
  "unresolved": []
}
```

### Partial or failure
```json
{
  "generated": false,
  "total_cycles": 200,
  "total_tokens": 84210,
  "assignments": [],
  "unresolved": [
    {
      "course_id": "CS201",
      "reason": "Abandoned after 5 retries. No lab available without lecturer conflict."
    }
  ]
}
```

---

## 13. main.py

```python
import asyncio
import json
from pydantic_ai import Agent
from control.coordinator import Coordinator
from core.data_loader import load_data
from core.deps import Deps
from core.logger import logger
from core.llm_factory import make_model
from store.store import Store
from agents.orchestrator_agent import OrchestratorAgent
from agents.course_agent import CourseAgent
from agents.room_agent import RoomAgent
from agents.lecturer_agent import LecturerAgent
from agents.policy_agent import PolicyAgent
from schemas.timetable import OrchestratorDecision, Proposal
from tools.agent_logger import log_decision

PROVIDER = "openai:gpt-4o"

async def main():
    courses, rooms, lecturers, policy = load_data()

    store = Store()
    store.seed([c.id for c in courses])

    deps = Deps(
        store=store,
        courses=courses,
        rooms=rooms,
        lecturers=lecturers,
        policy=policy,
        total_tokens=0,
    )

    agents = {
        "orchestrator": OrchestratorAgent(
            name="orchestrator",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=OrchestratorDecision, tools=[log_decision]),
        ),
        "course": CourseAgent(
            name="course",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        "room": RoomAgent(
            name="room",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        "lecturer": LecturerAgent(
            name="lecturer",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        "policy": PolicyAgent(
            name="policy",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
    }

    result = await run(deps, agents)
    print(json.dumps(result, indent=2))

asyncio.run(main())
```

Agent instantiation is explicit in `main.py` — the model, deps_type, output_type, and tools are all visible in one place. Each agent class receives a fully configured `pydantic_ai.Agent` and a name; it does not construct its own internally. Prompts — both instruction and user — are owned entirely by each agent class.

---

## 14. Implementation Sequence

Build and verify in this order. Each phase has a verification step — do not proceed to the next phase until verification passes.

**Phase 1 — Schemas** (`schemas/`)
All Pydantic models including `OrchestratorDecision`. Verify by instantiating each with dummy data and round-tripping through `model_dump()` / `model_validate()`.

**Phase 2 — Store** (`store/store.py`)
Seed, confirm, record_rejection, abandon, all three reads. Verify with a standalone script: seed three courses, confirm one, reject one, abandon one, print all collections.

**Phase 3 — Core infrastructure** (`core/`)
`deps.py`, `llm_factory.py`, `data_loader.py`, `logger.py`. Verify by loading data files and printing parsed models.

**Phase 4 — Data files** (`data/`)
Copy the four JSON files from section 11. Verify by running `load_data()` and confirming all models parse without error.

**Phase 5 — Tools** (`tools/agent_logger.py`)
The single `log_decision` tool. Verify by instantiating a minimal pydantic-ai agent with this tool and confirming a log line appears with a non-zero token count on first call.

**Phase 6 — Worker agents** (`agents/`)
Build in this order: `PolicyAgent` → `RoomAgent` → `LecturerAgent` → `CourseAgent`. For each: instantiate with a manually constructed `Agent`, call `run()` with dummy data, confirm the returned `Proposal` has the expected fields populated. Also confirm `get_instruction()` and `get_prompt()` produce non-empty strings with no missing data.

**Phase 7 — OrchestratorAgent** (`agents/orchestrator_agent.py`)
Build after all workers are verified. Test by constructing a realistic state snapshot (a few assignments, a rejection or two, an in-flight proposal at various stages) and confirming the returned `OrchestratorDecision.next_action` is correct for each scenario.

**Phase 8 — Control loop** (`control/coordinator.py`)
Assemble the full loop. Verify the happy path end-to-end: all four courses should confirm. Then verify failure recovery by temporarily making a constraint impossible (e.g. remove all lab rooms) and confirming the orchestrator abandons after MAX_RETRIES.

**Phase 9 — Integration** (`main.py`)
Run end-to-end. Observe log output to confirm: orchestrator is called every cycle, workers activate only when dispatched, `log_decision` lines appear with non-zero tokens, final output includes `total_tokens` reflecting cumulative cost.