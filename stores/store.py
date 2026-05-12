from schemas.timetable import Proposal, Assignment, RejectionRecord


class Store:
    """
    Persistence and audit layer for the scheduling run.

    Holds three collections:
        _assignments        — courses that have been fully confirmed
        _rejection_log      — every policy rejection recorded (used by the
                              orchestrator to count retries per course)
        _unscheduled_courses — course ids still waiting to be scheduled;
                              seeded at startup, shrinks as courses are
                              confirmed or abandoned

    Only the control loop (control/orchestrator.py) writes to the store.
    Workers never touch it. The orchestrator reads from it every cycle to
    build its state snapshot before making a decision.
    """

    def __init__(self):
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []

    # ── Seeding ────────────────────────────────────────────────────────────

    def seed(self, course_ids: list[str]) -> None:
        """
        Populates _unscheduled_courses with all course ids at startup.
        Called once in main.py before the control loop begins.
        Every course starts here and is removed only when confirmed or abandoned.
        """
        self._unscheduled_courses = list(course_ids)

    # ── Writes ─────────────────────────────────────────────────────────────

    def confirm(self, proposal: Proposal, cycle: int) -> None:
        """
        Called by the control loop when the orchestrator returns next_action='confirm'.
        Constructs an Assignment from the fully approved proposal, appends it to
        _assignments, and removes the course from _unscheduled_courses.

        At this point the proposal must have timeslot, room_id, lecturer_id all set
        and policy_approved=True — the orchestrator should never confirm otherwise.

        Raises ValueError if course_id is not in _unscheduled_courses, which would
        indicate a bug in the control loop (confirming something twice, or a course
        that was already abandoned).
        """
        assignment = Assignment(
            course_id=proposal.course_id,
            room_id=proposal.room_id,
            lecturer_id=proposal.lecturer_id,
            timeslot=proposal.timeslot,
            confirmed_at_cycle=cycle,
        )
        self._assignments.append(assignment)
        self._unscheduled_courses.remove(proposal.course_id)

    def record_rejection(self, course_id: str, reason: str, cycle: int) -> None:
        """
        Called by the control loop when the orchestrator returns a dispatch action
        after a policy rejection — i.e. the proposal failed and a retry is being
        attempted. Appends a RejectionRecord to the log.

        Does NOT remove the course from _unscheduled_courses — the course stays
        in play so the orchestrator can retry it. The orchestrator reads the
        rejection log each cycle to count how many times a course has been rejected
        and decides when to give up.
        """
        self._rejection_log.append(
            RejectionRecord(course_id=course_id, reason=reason, cycle=cycle)
        )

    def abandon(self, course_id: str) -> None:
        """
        Called by the control loop when the orchestrator returns next_action='abandon'.
        This happens when the rejection count for a course hits MAX_RETRIES and the
        orchestrator decides there is no point retrying further.

        Removes the course from _unscheduled_courses so the orchestrator stops
        seeing it as something still to be scheduled. The rejection log entries
        for this course remain as an audit trail and will appear in the final
        output under 'unresolved'.

        Raises ValueError if course_id is not found — same defensive behaviour
        as confirm().
        """
        self._unscheduled_courses.remove(course_id)

    # ── Reads ──────────────────────────────────────────────────────────────

    def get_assignments(self) -> list[Assignment]:
        """
        Returns all confirmed assignments so far.
        Called by the orchestrator every cycle to include in its state snapshot.
        Also called by worker agents (via deps) when building their prompts —
        CourseAgent uses it to see which timeslots are taken, RoomAgent and
        LecturerAgent use it to see what is already booked at a given slot.
        Returns a copy so callers cannot mutate internal state.
        """
        return list(self._assignments)

    def get_rejection_log(self) -> list[RejectionRecord]:
        """
        Returns the full rejection history.
        Called by the orchestrator every cycle to count retries per course
        and decide whether to keep retrying or abandon.
        Returns a copy so callers cannot mutate internal state.
        """
        return list(self._rejection_log)

    def get_unscheduled_courses(self) -> list[str]:
        """
        Returns course ids not yet confirmed or abandoned.
        Called by the orchestrator every cycle — when this is empty the
        orchestrator returns next_action='done' and the loop exits.
        Also used as the while condition in the control loop as a quick
        termination check before calling the orchestrator.
        Returns a copy so callers cannot mutate internal state.
        """
        return list(self._unscheduled_courses)