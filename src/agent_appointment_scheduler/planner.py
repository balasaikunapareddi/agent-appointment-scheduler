"""Plan-then-act.

For a multi-step booking the agent forms an explicit short plan first,
then executes it step by step. The plan is deterministic data — a list of
steps with the tool each step uses — built from the booking stage, not
improvised by the model. When a tool call fails, the agent replans from
the failed step instead of improvising a new path.

Plans are exposed to the agent as the ``get_plan`` MCP tool and to the
demo, which prints the plan before executing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One step of a plan: which tool, and what it needs."""

    step: int
    tool: str
    purpose: str
    done: bool = False


class Plan(BaseModel):
    """A short, explicit plan for the current booking stage."""

    goal: str
    steps: list[PlanStep] = Field(default_factory=list)

    def next_step(self) -> PlanStep | None:
        for step in self.steps:
            if not step.done:
                return step
        return None

    def mark_done(self, tool: str) -> None:
        for step in self.steps:
            if step.tool == tool and not step.done:
                step.done = True
                return

    def readable(self) -> str:
        lines = [f"Plan: {self.goal}"]
        for step in self.steps:
            mark = "done" if step.done else "next" if step is self.next_step() else "todo"
            lines.append(f"  {step.step}. [{mark}] {step.tool} — {step.purpose}")
        return "\n".join(lines)


def plan_for(stage: str, goal: str = "book an appointment") -> Plan:
    """Build the plan for a booking stage. Pure function of the stage."""
    if stage in ("collecting", "providers_shown"):
        steps = [
            PlanStep(step=1, tool="search_providers", purpose="Find providers matching the need."),
            PlanStep(step=2, tool="get_availability", purpose="Show open slots for the chosen provider."),
            PlanStep(step=3, tool="hold_slot", purpose="Hold the chosen slot while the user confirms."),
            PlanStep(step=4, tool="get_booking_status", purpose="Show the masked summary and ask to confirm."),
            PlanStep(step=5, tool="book_appointment", purpose="Book with confirmation and idempotency key."),
        ]
    elif stage == "slot_picked":
        steps = [
            PlanStep(step=1, tool="get_booking_status", purpose="Show the masked summary and ask to confirm."),
            PlanStep(step=2, tool="book_appointment", purpose="Book with confirmation and idempotency key."),
        ]
    elif stage == "confirming":
        steps = [
            PlanStep(step=1, tool="book_appointment", purpose="Book with confirmation and idempotency key."),
        ]
    elif stage == "booked":
        steps = [
            PlanStep(step=1, tool="list_appointments", purpose="Show the booked appointment."),
        ]
    else:
        steps = [
            PlanStep(step=1, tool="search_providers", purpose="Find providers matching the need."),
        ]
    return Plan(goal=goal, steps=steps)


def replan(plan: Plan, failed_tool: str, error: str) -> Plan:
    """Rebuild the plan after a tool failure, from the failed step on.

    Deterministic: the failed step is retried once with a fallback note,
    and steps after it stay queued. The agent never invents a new path.
    """
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.tool == failed_tool and not step.done:
            steps.append(
                PlanStep(
                    step=len(steps) + 1,
                    tool=step.tool,
                    purpose=f"{step.purpose} (retry after failure: {error})",
                )
            )
        elif not step.done:
            steps.append(
                PlanStep(step=len(steps) + 1, tool=step.tool, purpose=step.purpose)
            )
        else:
            steps.append(
                PlanStep(step=len(steps) + 1, tool=step.tool, purpose=step.purpose, done=True)
            )
    return Plan(goal=f"{plan.goal} (replanned after {failed_tool} failed)", steps=steps)
