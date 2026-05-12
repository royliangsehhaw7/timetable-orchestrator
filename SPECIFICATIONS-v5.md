# Multi-Agent Timetable Generation System
## Specification v6.0

---

## 1. Purpose

A learning-oriented multi-agent system that generates a valid weekly school timetable. The primary goal is to understand and implement two complementary design patterns — the **Orchestrator-Worker Pattern** and the **Reflexion Pattern** — in a grounded, practical context using pydantic-ai's deps, agents, and tools features properly.

The timetable domain is intentionally simple. It exists to make data flow visible and traceable, not to solve a hard scheduling problem.

---

## 2. Architecture

### 2.1 Pattern

This system is a **hybrid**: a Pipeline data flow within an Orchestrator-Worker control structure.

The per-course scheduling sequence is a pipeline by domain necessity — a room cannot be assigned before a timeslot exists, a lecturer cannot be assigned before a room exists. This ordering is enforced by data dependencies, not by the pattern.

The Orchestrator-Worker pattern sits above this pipeline. One orchestrator agent owns all control decisions: what to schedule next, when to advance a proposal to the next stage, how to recover from failure, and when to give up on a course. Worker agents are domain specialists who do deep reasoning within their area and return structured results. Workers never call each other. Only the orchestrator dispatches workers.

### 2.2 Agents

There are five pydantic-ai agents:

| Agent | Role |
|-------|------|
| `OrchestratorAgent` | Strategic control — picks courses, dispatches workers, handles failure routing |
| `CourseAgent` | Timeslot domain specialist |
| `RoomAgent` | Room fit specialist |
| `LecturerAgent` | Lecturer suitability specialist |
| `PolicyAgent` | Compliance specialist |

All five share a common base class. The orchestrator is not a dumb loop — it is an LLM-powered agent that reasons over failures and multi-course strategy. Workers are LLM-powered agents that reason deeply within their domain.

### 2.3 Flow

**Happy path** — handoffs managed in Python after each worker returns:

```
OrchestratorAgent
    picks next unscheduled course
        → CourseAgent
            reasons about best timeslot
            returns Proposal with timeslot set
        Python: field present? → pass forward
        → RoomAgent
            reasons about best room fit
            returns Proposal with room_id set
        Python: field present? → pass forward
        → LecturerAgent
            reasons about best lecturer
            returns Proposal with lecturer_id set
        Python: field present? → pass forward
        → PolicyAgent
            deep compliance check
            returns Proposal with policy_approved + policy_reason
        Python: approved? → confirm to store
```

**Failure path** — OrchestratorAgent LLM reasoning activated:

```
PolicyAgent returns policy_approved=False with specific reason
    → OrchestratorAgent reasons over rejection reason
    → identifies which specialist is responsible
    → re-dispatches that specialist with failure context
    → if retry limit hit → flag course, move to next
```

**Between courses** — OrchestratorAgent LLM reasoning activated:

```
OrchestratorAgent inspects store
    → decides which course to tackle next
    → considers: confirmed assignments, rejection history, dependencies
```

### 2.4 What the Orchestrator does not do

The orchestrator does not perform domain reasoning. It does not choose timeslots, rooms, or lecturers. It does not evaluate policy rules. It reads outcomes, makes routing decisions, and manages the lifecycle of each proposal. Domain judgment belongs entirely to the specialist workers.

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
│   └── orchestrator.py
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
    rejection_reason: str | None = None
    retry_count: int = 0

class Assignment(BaseModel):
    course_id: str
    room_id: str
    lecturer_id: str
    timeslot: TimeSlot
    confirmed_at_cycle: int

class RejectionRecord(BaseModel):
    course_id: str
    reason: str
    cycle: int
```

The `Proposal` no longer carries a `status` field. Status is implicit: a proposal is in-flight while it lives in the orchestrator's working memory. Once the orchestrator is done with it, it either becomes an `Assignment` in the store (confirmed) or a `RejectionRecord` (abandoned). The orchestrator holds the active proposal directly as a local variable, not in the store.

**Verification**: import each schema in a scratch script and instantiate one instance per model with dummy data. All fields should round-trip through `model_dump()` and `model_validate()` without error.

---

## 5. The Store

**Location**: `store/store.py`

The store is a persistence and audit layer for the orchestrator. It records outcomes — confirmed assignments and rejection history — and provides availability data that the orchestrator injects into worker prompts. It is not a coordination medium between agents. In-flight proposals live in the orchestrator's working memory, not here.

### 5.1 What it holds

```python
class Store:
    def __init__(self):
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []
```

`_unscheduled_courses` is seeded at startup. A course id is removed only when its proposal is confirmed as an assignment.

### 5.2 Methods

#### Seeding

| Method | Signature | What it does |
|--------|-----------|--------------|
| `seed` | `(course_ids: list[str]) -> None` | Populates `_unscheduled_courses` at startup. Called once before the control loop begins. |

#### Write methods

| Method | Signature | What it does |
|--------|-----------|--------------|
| `confirm` | `(proposal: Proposal, cycle: int) -> None` | Constructs an `Assignment` from the proposal, appends to `_assignments`, removes `course_id` from `_unscheduled_courses`. |
| `record_rejection` | `(course_id: str, reason: str, cycle: int) -> None` | Appends a `RejectionRecord`. Course id remains in `_unscheduled_courses` unless the orchestrator decides to abandon it, in which case it calls `abandon`. |
| `abandon` | `(course_id: str) -> None` | Removes `course_id` from `_unscheduled_courses`. Called by the orchestrator when retry limit is reached. |

#### Read methods

| Method | Signature | What it returns |
|--------|-----------|-----------------|
| `get_assignments` | `() -> list[Assignment]` | All confirmed assignments. Used by the orchestrator to build availability context for worker prompts. |
| `get_rejection_log` | `() -> list[RejectionRecord]` | Full rejection history. Used by the orchestrator to count retries per course. |
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
    store: Store                 # mutable — orchestrator writes to this
    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    total_tokens: int = 0        # accumulated token usage across all agents
```

The store is the same instance throughout the entire run. Workers do not write to the store — only the orchestrator does. Workers receive their context in the prompt and return structured results via `output_type`.

### 6.2 How Deps flows at runtime

The orchestrator creates `Deps` once before the control loop starts:

```python
deps = Deps(
    store=store,
    courses=courses,
    rooms=rooms,
    lecturers=lecturers,
    policy=policy,
    total_tokens=0,
)
```

The same `deps` instance is passed into every agent's `run()` call. pydantic-ai injects it into `RunContext` so the `log_decision` tool can accumulate token usage into `deps.total_tokens`.

---

## 7. Tools

### 7.1 Design decision

There is one tool in this system: `log_decision`. It is registered on every agent including the orchestrator.

A tool is something the LLM calls mid-reasoning to perform an action with a side effect. `log_decision` qualifies: the LLM triggers it deliberately to narrate its reasoning, and it produces structured log output and accumulates token counts as side effects.

Store writes (confirm, record_rejection, abandon) happen in Python inside the control loop after the LLM returns its structured result. They are not tools because the LLM does not need to trigger them mid-reasoning — the Python code acts on the returned `output_type`.

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

Each agent has one responsibility. Each agent is powered by an LLM. The LLM makes the domain decision — which timeslot to propose, which room fits, which lecturer is available, whether policy is satisfied. Python does not make these decisions.

Each agent has a single `build_prompt()` method that returns one complete f-string: role, data, and rules combined. There is no system prompt. There is no separate task brief. Workers return `Proposal` with the relevant fields filled in. The orchestrator returns `OrchestratorRoutingDecision`.

Workers do not read from the store directly. They do not call each other.

### 8.2 BaseAgent

**Location**: `agents/base_agent.py`

```python
class BaseAgent:
    name: str
    agent: pydantic_ai.Agent

    def _with_failure(self, prompt: str, failure_reason: str | None) -> str:
        if failure_reason is None:
            return prompt
        return (
            prompt
            + "\n\n---\nCORRECTION REQUIRED\n"
            + "Your previous result was rejected for the following reason:\n"
            + failure_reason
            + "\n\nReturn a different result that avoids this problem.\n---"
        )
```

Each agent overrides `run` with its own typed arguments and calls its own `build_prompt` internally. The control loop calls `agent.run(...)` with domain args only — no prompt string is ever passed from outside.

**pydantic-ai agent instantiation** — each agent defines this in its own `__init__`:

```python
self.agent = pydantic_ai.Agent(
    model=make_model(provider),
    deps_type=Deps,
    output_type=...,
    tools=[log_decision],
)
```

No `system_prompt`. Each agent defines its own `output_type` and `build_prompt()`.

---

### 8.3 CourseAgent

- **Location**: `agents/course_agent.py`
- **Responsibility**: Given a course and full scheduling context, propose the best timeslot.
- **Output type**: `Proposal`

**Run**:

```python
async def run(self, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
    result = await self.agent.run(
        self._with_failure(self.build_prompt(course, deps), failure_reason), deps=deps
    )
    return result.data
```

**Prompt**:

```python
def build_prompt(self, course: Course, deps: Deps) -> str:
    assignments = deps.store.get_assignments()
    return f"""
        You are the CourseAgent in a timetable scheduling system. Propose the best timeslot for this course. Return the full proposal with timeslot set.

        Course:
        {json.dumps(course.model_dump(), indent=2)}

        School policy:
        {json.dumps(deps.policy.model_dump(), indent=2)}

        Already confirmed assignments (timeslots already taken):
        {json.dumps([a.model_dump() for a in assignments], indent=2)}

        Rules:
        - The timeslot must be on a valid school day, within school hours, and not during the lunch break
        - Avoid timeslots already taken by other courses
        - Prefer to spread courses across the week rather than clustering them
        - Consider the course type — lab courses benefit from longer uninterrupted blocks

        Call log_decision to explain your reasoning. Then return the proposal with timeslot set.
    """
```

---

### 8.4 RoomAgent

- **Location**: `agents/room_agent.py`
- **Responsibility**: Given a proposal with a timeslot, assign the most suitable room.
- **Output type**: `Proposal`

**Run**:

```python
async def run(self, proposal: Proposal, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
    result = await self.agent.run(
        self._with_failure(self.build_prompt(proposal, course, deps), failure_reason), deps=deps
    )
    return result.data
```

**Prompt**:

```python
def build_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
    assignments = deps.store.get_assignments()
    booked_at_slot = [
        a for a in assignments
        if a.timeslot.day == proposal.timeslot.day
        and a.timeslot.start_hour == proposal.timeslot.start_hour
    ]
    return f"""
        You are the RoomAgent in a timetable scheduling system. Assign the most suitable room for this course. Return the full proposal with room_id set.

        Current proposal:
        {json.dumps(proposal.model_dump(), indent=2)}

        Course:
        {json.dumps(course.model_dump(), indent=2)}

        All rooms:
        {json.dumps([r.model_dump() for r in deps.rooms], indent=2)}

        Rooms already confirmed at this timeslot:
        {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}

        Rules:
        - The room type must match the course requirement (lab course needs a lab room, non-lab course needs a classroom)
        - The room must not already be confirmed at this timeslot
        - If multiple suitable rooms are free, prefer the one that best fits the course

        Call log_decision to explain your reasoning. Then return the proposal with room_id set.
    """
```

---

### 8.5 LecturerAgent

- **Location**: `agents/lecturer_agent.py`
- **Responsibility**: Given a proposal with a timeslot and room, assign the most suitable lecturer.
- **Output type**: `Proposal`

**Run**:

```python
async def run(self, proposal: Proposal, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
    result = await self.agent.run(
        self._with_failure(self.build_prompt(proposal, course, deps), failure_reason), deps=deps
    )
    return result.data
```

**Prompt**:

```python
def build_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
    assignments = deps.store.get_assignments()
    booked_at_slot = [
        a for a in assignments
        if a.timeslot.day == proposal.timeslot.day
        and a.timeslot.start_hour == proposal.timeslot.start_hour
    ]
    return f"""
        You are the LecturerAgent in a timetable scheduling system. Assign the most suitable lecturer for this course. Return the full proposal with lecturer_id set.

        Current proposal:
        {json.dumps(proposal.model_dump(), indent=2)}

        Course:
        {json.dumps(course.model_dump(), indent=2)}

        All lecturers (includes courses_taught and unavailable_slots):
        {json.dumps([l.model_dump() for l in deps.lecturers], indent=2)}

        Lecturers already confirmed at this timeslot:
        {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}

        Rules:
        - The lecturer must teach this course (check courses_taught)
        - The lecturer must not already be confirmed at this timeslot
        - The lecturer must not have this timeslot in their unavailable_slots
        - If multiple qualified lecturers are free, prefer the one with the lighter confirmed workload so far

        Call log_decision to explain your reasoning. Then return the proposal with lecturer_id set.
    """
```

---

### 8.6 PolicyAgent

- **Location**: `agents/policy_agent.py`
- **Responsibility**: Given a fully assembled proposal, perform a deep compliance check against school policy.
- **Output type**: `Proposal`

**Run**:

```python
async def run(self, proposal: Proposal, course: Course, deps: Deps, failure_reason: str | None = None) -> Any:
    result = await self.agent.run(
        self._with_failure(self.build_prompt(proposal, course, deps), failure_reason), deps=deps
    )
    return result.data
```

**Prompt**:

```python
def build_prompt(self, proposal: Proposal, course: Course, deps: Deps) -> str:
    room = next(r for r in deps.rooms if r.id == proposal.room_id)
    lecturer = next(l for l in deps.lecturers if l.id == proposal.lecturer_id)
    return f"""
        You are the PolicyAgent in a timetable scheduling system. Check this proposal against all school policy rules. Return the full proposal with policy_approved and policy_reason set.

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

        Rules:
        - The day must be a valid school day
        - The timeslot must fall entirely within school hours (start and end)
        - The timeslot must not overlap with the lunch break
        - The room type must match the course requirement
        - The lecturer must be qualified to teach this course
        - The lecturer must not be unavailable at this timeslot

        If all rules pass, set policy_approved=True. If any rule fails, set policy_approved=False and policy_reason to a specific actionable reason identifying exactly which rule was violated.

        Call log_decision to explain your assessment. Then return the proposal with policy_approved and policy_reason set.
    """
```

---

### 8.7 OrchestratorAgent

- **Location**: `agents/orchestrator_agent.py`
- **Responsibility**: Route failures to the correct specialist.
- **Output type**:

```python
class OrchestratorRoutingDecision(BaseModel):
    dispatch_to: str   # "course" | "room" | "lecturer" | "abandon"
    reason: str
```

**Run**:

```python
async def run(self, proposal: Proposal, rejection_reason: str, retry_count: int, deps: Deps) -> Any:
    result = await self.agent.run(
        self.build_prompt(proposal, rejection_reason, retry_count), deps=deps
    )
    return result.data
```

**Prompt**:

```python
def build_prompt(self, proposal: Proposal, rejection_reason: str, retry_count: int) -> str:
    return f"""
You are the OrchestratorAgent in a timetable scheduling system. A scheduling proposal has been rejected. Decide which specialist agent should correct it.

Rejected proposal:
{json.dumps(proposal.model_dump(), indent=2)}

Rejection reason:
{rejection_reason}

Retry information:
attempts so far: {retry_count}
maximum allowed: {MAX_RETRIES}

Rules:
- Timeslot-related violations → dispatch_to: "course"
- Room type or availability violations → dispatch_to: "room"
- Lecturer qualification or availability violations → dispatch_to: "lecturer"
- If the course has exceeded the retry limit → dispatch_to: "abandon"

Call log_decision to explain your routing decision. Then return dispatch_to and a clear reason that will be passed to the specialist as their failure context.
    """
```

---

## 9. Control Flow

**Location**: `control/orchestrator.py`

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

```python
MAX_CYCLES = 200
MAX_RETRIES = 5

async def run(deps: Deps, agents: dict) -> dict:
    cycle = 0

    while deps.store.get_unscheduled_courses():
        cycle += 1

        if cycle > MAX_CYCLES:
            break

        logger.info(f"[cycle {cycle}] unscheduled: {deps.store.get_unscheduled_courses()}")

        course_id = deps.store.get_unscheduled_courses()[0]
        course = next(c for c in deps.courses if c.id == course_id)

        proposal = Proposal(id=str(uuid.uuid4()), course_id=course_id)
        retry_count = sum(
            1 for r in deps.store.get_rejection_log()
            if r.course_id == course_id
        )

        if retry_count >= MAX_RETRIES:
            deps.store.abandon(course_id)
            logger.warning(f"[cycle {cycle}] abandoning {course_id} after {retry_count} retries")
            continue

        failure_reason = None

        # Stage 1: timeslot
        result = await agents["course"].run(
            course=course,
            deps=deps,
            failure_reason=failure_reason,
        )
        if not result.timeslot:
            logger.warning(f"[cycle {cycle}] CourseAgent returned no timeslot for {course_id}")
            continue
        proposal.timeslot = result.timeslot

        # Stage 2: room
        result = await agents["room"].run(
            proposal=proposal,
            course=course,
            deps=deps,
        )
        if not result.room_id:
            logger.warning(f"[cycle {cycle}] RoomAgent returned no room for {course_id}")
            continue
        proposal.room_id = result.room_id

        # Stage 3: lecturer
        result = await agents["lecturer"].run(
            proposal=proposal,
            course=course,
            deps=deps,
        )
        if not result.lecturer_id:
            logger.warning(f"[cycle {cycle}] LecturerAgent returned no lecturer for {course_id}")
            continue
        proposal.lecturer_id = result.lecturer_id

        # Stage 4: policy
        result = await agents["policy"].run(
            proposal=proposal,
            course=course,
            deps=deps,
        )
        proposal.policy_approved = result.policy_approved
        proposal.policy_reason = result.policy_reason

        if result.policy_approved:
            deps.store.confirm(proposal, cycle)
            logger.info(f"[cycle {cycle}] confirmed {course_id}")
        else:
            deps.store.record_rejection(course_id, result.policy_reason, cycle)
            proposal.retry_count += 1

            routing = await agents["orchestrator"].run(
                proposal=proposal,
                rejection_reason=result.policy_reason,
                retry_count=proposal.retry_count,
                deps=deps,
            )
            logger.info(
                f"[cycle {cycle}] rejection routed to {routing.dispatch_to}: {routing.reason}"
            )

    return _produce_output(deps, cycle)
```

### 9.3 Termination

| Condition | Outcome |
|-----------|---------|
| `get_unscheduled_courses()` empty | Success — full timetable produced |
| Course retry count reaches `MAX_RETRIES` | Abandoned — store.abandon called, move to next course |
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
from control.orchestrator import run
from core.data_loader import load_data
from core.deps import Deps
from core.logger import logger
from store.store import Store
from agents.orchestrator_agent import OrchestratorAgent
from agents.course_agent import CourseAgent
from agents.room_agent import RoomAgent
from agents.lecturer_agent import LecturerAgent
from agents.policy_agent import PolicyAgent

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
        "orchestrator": OrchestratorAgent(PROVIDER),
        "course":       CourseAgent(PROVIDER),
        "room":         RoomAgent(PROVIDER),
        "lecturer":     LecturerAgent(PROVIDER),
        "policy":       PolicyAgent(PROVIDER),
    }

    result = await run(deps, agents)
    print(json.dumps(result, indent=2))

asyncio.run(main())
```

---

## 14. Implementation Sequence

Build and verify in this order. Each phase has a verification step — do not proceed to the next phase until verification passes.

**Phase 1 — Schemas** (`schemas/`)
All Pydantic models. Verify by instantiating each with dummy data and round-tripping through `model_dump()` / `model_validate()`.

**Phase 2 — Store** (`store/store.py`)
Seed, confirm, record_rejection, abandon, all three reads. Verify with a standalone script: seed three courses, confirm one, reject one, abandon one, print all collections.

**Phase 3 — Core infrastructure** (`core/`)
`deps.py`, `llm_factory.py`, `data_loader.py`, `logger.py`. Verify by loading data files and printing parsed models.

**Phase 4 — Data files** (`data/`)
Copy the four JSON files from section 11. Verify by running `load_data()` and confirming all models parse without error.

**Phase 5 — Tools** (`tools/agent_logger.py`)
The single `log_decision` tool. Verify by instantiating a minimal pydantic-ai agent with this tool and confirming a log line appears with a non-zero token count on first call.

**Phase 6 — Worker agents** (`agents/`)
Build in this order: `PolicyAgent` → `RoomAgent` → `LecturerAgent` → `CourseAgent`. PolicyAgent first because its logic is most deterministic and easiest to verify. For each: instantiate with a real `Deps`, call `run(prompt=agent.build_prompt(...), deps=deps)` with a manually constructed prompt containing the relevant serialised data, confirm the returned `Proposal` has the expected fields populated.

**Phase 7 — OrchestratorAgent** (`agents/orchestrator_agent.py`)
Build after all workers are verified. Test failure routing by constructing a rejected proposal and a rejection reason, calling `run(prompt=agent.build_prompt(...), deps=deps)`, and confirming the returned `dispatch_to` correctly identifies the responsible specialist.

**Phase 8 — Control loop** (`control/orchestrator.py`)
Assemble the full loop. Verify the happy path end-to-end: all four courses should confirm within a small number of cycles. Then verify failure recovery by temporarily making a constraint impossible (e.g. remove all lab rooms) and confirming the orchestrator abandons the course after MAX_RETRIES.

**Phase 9 — Integration** (`main.py`)
Run end-to-end. Observe log output to confirm: CourseAgent acts first per course, workers activate in pipeline order, `log_decision` lines appear with non-zero tokens, final output includes `total_tokens` reflecting cumulative cost.