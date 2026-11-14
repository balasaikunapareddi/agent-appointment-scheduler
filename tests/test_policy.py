"""Policy + planner tests: guardrails in code, not in the prompt."""

from agent_appointment_scheduler import store
from agent_appointment_scheduler.planner import Plan, plan_for, replan
from agent_appointment_scheduler.policy import check_book, check_cancel, check_reschedule


def test_plan_for_stage():
    plan = plan_for("collecting")
    assert plan.steps[0].tool == "search_providers"
    assert plan.steps[-1].tool == "book_appointment"
    assert plan_for("booked").steps[-1].tool == "list_appointments"
    assert plan_for("slot_picked").steps[0].tool == "get_booking_status"


def test_replan_retries_failed_step_once():
    plan = plan_for("collecting")
    replanned = replan(plan, failed_tool="get_availability", error="timeout")
    assert replanned.next_step().tool == "search_providers"  # first undone step is still first
    failed = next(s for s in replanned.steps if s.tool == "get_availability")
    assert "retry after failure: timeout" in failed.purpose


def test_check_book_requires_confirmation(db_path):
    conn = store.connect(db_path)
    try:
        decision = check_book(conn, confirmed=False, idempotency_key="k", npi="1", slot="2026-11-18T09:00")
    finally:
        conn.close()
    assert not decision.allowed
    assert "confirm" in decision.reason.lower()


def test_check_book_requires_idempotency_key(db_path):
    conn = store.connect(db_path)
    try:
        decision = check_book(conn, confirmed=True, idempotency_key="", npi="1", slot="2026-11-18T09:00")
    finally:
        conn.close()
    assert not decision.allowed


def test_check_book_refuses_taken_slot(db_path, booked):
    conn = store.connect(db_path)
    try:
        decision = check_book(
            conn, confirmed=True, idempotency_key="k2", npi=booked["npi"], slot=booked["slot"]
        )
    finally:
        conn.close()
    assert not decision.allowed
    assert "already booked" in decision.reason


def test_check_book_refuses_expired_hold(db_path, client, provider_and_slot):
    provider, slot = provider_and_slot
    hold = client.post("/holds", json={"npi": provider["npi"], "slot": slot, "session_id": "SES-P1"}).json()
    conn = store.connect(db_path)
    try:
        import time

        conn.execute("UPDATE holds SET expires_at = ? WHERE hold_id = ?", (time.time() - 60, hold["hold_id"]))
        conn.commit()
        decision = check_book(
            conn, confirmed=True, idempotency_key="k3", npi=provider["npi"], slot=slot, hold_id=hold["hold_id"]
        )
    finally:
        conn.close()
    assert not decision.allowed
    assert "expired" in decision.reason.lower()


def test_check_book_allows(db_path, provider_and_slot):
    provider, slot = provider_and_slot
    conn = store.connect(db_path)
    try:
        decision = check_book(conn, confirmed=True, idempotency_key="k4", npi=provider["npi"], slot=slot)
    finally:
        conn.close()
    assert decision.allowed


def test_check_cancel_and_reschedule(db_path, booked, client, provider_and_slot):
    provider, slot = provider_and_slot
    conn = store.connect(db_path)
    try:
        assert check_cancel(conn, appointment_id=booked["appointment_id"]).allowed
        availability = client.get(f"/providers/{provider['npi']}/availability").json()
        other_slot = next(s["slot"] for s in availability["slots"] if s["available"] and s["slot"] != slot)
        assert check_reschedule(
            conn, appointment_id=booked["appointment_id"], npi=provider["npi"], new_slot=other_slot
        ).allowed
        # Rescheduling onto its own slot is fine (excluded from the taken check).
        assert check_reschedule(
            conn, appointment_id=booked["appointment_id"], npi=provider["npi"], new_slot=slot
        ).allowed
    finally:
        conn.close()
    client.post(f"/appointments/{booked['appointment_id']}/cancel", json={})
    conn = store.connect(db_path)
    try:
        assert not check_cancel(conn, appointment_id=booked["appointment_id"]).allowed
        assert not check_reschedule(
            conn, appointment_id=booked["appointment_id"], npi=provider["npi"], new_slot=other_slot
        ).allowed
    finally:
        conn.close()


def test_plan_serializes():
    plan = plan_for("slot_picked")
    dumped = plan.model_dump()
    assert Plan(**dumped).readable()
