"""Policy-as-code guardrails.

The agent proposes; the policy layer disposes. Every consequential action
(book / cancel / reschedule) passes a deterministic check here BEFORE it
happens — enforced in code in the FastAPI service and again in the MCP
tools, never left to the system prompt. This is the same shape as Bala's
agent-policy-gateway: a small, auditable policy engine the LLM cannot
talk its way around.

Checks, in order:
- confirmation: a booking needs an explicit user confirmation.
- idempotency: a booking needs an idempotency key.
- no double-booking: the slot must be free at booking time.
- hold validity: a held slot must exist and not have expired.
- appointment exists and is in a state the action allows.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

HOLD_TTL_SECONDS = 10 * 60  # a slot hold lasts 10 minutes


@dataclass
class PolicyDecision:
    allowed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.allowed


def _appointment(conn: sqlite3.Connection, appointment_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
    ).fetchone()
    return dict(row) if row else None


def _slot_taken(conn: sqlite3.Connection, npi: str, slot: str, exclude: str | None = None) -> bool:
    query = (
        "SELECT appointment_id FROM appointments "
        "WHERE npi = ? AND slot = ? AND status = 'booked'"
    )
    params: list[str] = [npi, slot]
    if exclude:
        query += " AND appointment_id != ?"
        params.append(exclude)
    return conn.execute(query, params).fetchone() is not None


def _hold_valid(conn: sqlite3.Connection, hold_id: str, npi: str, slot: str, now: float) -> PolicyDecision:
    row = conn.execute("SELECT * FROM holds WHERE hold_id = ?", (hold_id,)).fetchone()
    if row is None:
        return PolicyDecision(False, f"Hold {hold_id} does not exist.")
    hold = dict(row)
    if hold["npi"] != npi or hold["slot"] != slot:
        return PolicyDecision(False, "Hold does not match this provider and slot.")
    if hold["expires_at"] <= now:
        return PolicyDecision(False, "Hold has expired; pick the slot again.")
    return PolicyDecision(True, "Hold is valid.")


def check_book(
    conn: sqlite3.Connection,
    *,
    confirmed: bool,
    idempotency_key: str,
    npi: str,
    slot: str,
    hold_id: str | None = None,
    now: float | None = None,
) -> PolicyDecision:
    """Gate a booking. Every check is deterministic; no LLM judgement."""
    now = now if now is not None else time.time()
    if not confirmed:
        return PolicyDecision(
            False,
            "Confirmation required: the user must explicitly confirm the "
            "booking before it is submitted.",
        )
    if not (idempotency_key or "").strip():
        return PolicyDecision(False, "An idempotency key is required to book.")
    if _slot_taken(conn, npi, slot):
        return PolicyDecision(False, "That slot is already booked.")
    if hold_id:
        decision = _hold_valid(conn, hold_id, npi, slot, now)
        if not decision:
            return decision
    return PolicyDecision(True, "Booking allowed.")


def check_cancel(conn: sqlite3.Connection, *, appointment_id: str) -> PolicyDecision:
    appt = _appointment(conn, appointment_id)
    if appt is None:
        return PolicyDecision(False, f"Appointment {appointment_id} does not exist.")
    if appt["status"] == "cancelled":
        return PolicyDecision(False, f"Appointment {appointment_id} is already cancelled.")
    return PolicyDecision(True, "Cancellation allowed.")


def check_reschedule(
    conn: sqlite3.Connection, *, appointment_id: str, npi: str, new_slot: str
) -> PolicyDecision:
    appt = _appointment(conn, appointment_id)
    if appt is None:
        return PolicyDecision(False, f"Appointment {appointment_id} does not exist.")
    if appt["status"] == "cancelled":
        return PolicyDecision(False, "A cancelled appointment cannot be rescheduled.")
    if _slot_taken(conn, npi, new_slot, exclude=appointment_id):
        return PolicyDecision(False, "The new slot is already booked.")
    return PolicyDecision(True, "Reschedule allowed.")
