"""Deterministic eval suite (the agent-eval-lite approach).

Fixed scenarios, fixed assertions, no model key needed — everything
runs on the scripted model plus direct API calls. Run with ``make eval``.

Scenarios:
1. find + book — the full happy path, end to end.
2. book + reschedule — same session, same appointment id, new slot.
3. correction mid-flow — phone corrected after the patient record exists.
4. double-book refusal — a second booking for the same slot is refused.
5. expired hold — booking against an expired hold is refused.

Each scenario reports pass/fail with the assertion that failed, and the
run exits non-zero when any scenario fails.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import time

os.environ.setdefault("SCHEDULER_API_TRANSPORT", "asgi")

from . import store
from .chat import ChatService


def _fresh_service() -> tuple[ChatService, str]:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = tmp.name
    os.environ["SCHEDULER_DB_PATH"] = db_path
    store.connect(db_path).close()
    return ChatService(db_path=db_path, mock=True), db_path


def _book_flow(service: ChatService, session_id: str, phone: str = "612-555-0147") -> dict:
    """Run the standard booking turns. Returns the last reply + session id."""
    turns = [
        "I need a cardiologist near Minneapolis next week, I prefer mornings",
        "the second one",
        "Wednesday morning works",
        f"My name is Jordan Lee, born May 14, 1990, phone {phone}",
        "It's a heart checkup follow-up, text me",
        "Yes, book it",
    ]
    reply = None
    for turn in turns:
        reply = service.chat(session_id, turn)
    return reply


def _appointments(db_path: str) -> list[dict]:
    conn = store.connect(db_path)
    try:
        rows = conn.execute("SELECT * FROM appointments").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def scenario_find_and_book() -> str:
    service, db_path = _fresh_service()
    reply = _book_flow(service, "SES-EVAL1")
    appointments = _appointments(db_path)
    assert len(appointments) == 1, f"expected 1 appointment, got {len(appointments)}"
    assert "Booked!" in reply.reply, f"expected a booking confirmation, got: {reply.reply[:200]}"
    assert appointments[0]["appointment_id"].startswith("APT-"), "bad appointment id"
    assert reply.booking_stage == "booked", f"stage is {reply.booking_stage}"
    # PHI: the raw phone must not appear in stored history.
    conn = store.connect(db_path)
    try:
        rows = conn.execute("SELECT content FROM session_messages").fetchall()
    finally:
        conn.close()
    for row in rows:
        assert "612-555-0147" not in row["content"], "raw phone leaked into stored history"
        assert "Jordan Lee" not in row["content"], "raw name leaked into stored history"
    service.close()
    return f"booked {appointments[0]['appointment_id']} on {appointments[0]['slot']}"


def scenario_book_and_reschedule() -> str:
    service, db_path = _fresh_service()
    _book_flow(service, "SES-EVAL2")
    before = _appointments(db_path)[0]
    reply = service.chat("SES-EVAL2", "Actually, can we move it to Thursday morning instead?")
    after = _appointments(db_path)[0]
    assert after["appointment_id"] == before["appointment_id"], "reschedule changed the appointment id"
    assert after["slot"] != before["slot"], "slot did not change"
    assert "Thursday" in after["slot"] or after["slot"][5:7] == "11", f"unexpected new slot {after['slot']}"
    assert "Moved it" in reply.reply, f"expected a reschedule confirmation, got: {reply.reply[:200]}"
    service.close()
    return f"{before['appointment_id']} moved {before['slot']} -> {after['slot']}"


def scenario_correction_mid_flow() -> str:
    service, db_path = _fresh_service()
    turns = [
        "I need a cardiologist near Minneapolis next week, I prefer mornings",
        "the second one",
        "Wednesday morning works",
        "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147",
        "Actually my phone is 612-555-0199",
    ]
    for turn in turns:
        reply = service.chat("SES-EVAL3", turn)
    assert "***-***-0199" in reply.reply or "0199" in reply.reply, f"correction not reflected: {reply.reply[:200]}"
    reply = service.chat("SES-EVAL3", "It's a heart checkup follow-up, text me")
    reply = service.chat("SES-EVAL3", "Yes, book it")
    assert "Booked!" in reply.reply, f"booking failed after correction: {reply.reply[:200]}"
    service.close()
    return "phone corrected to ***-***-0199, booking completed"


def scenario_double_book_refused() -> str:
    service, db_path = _fresh_service()
    _book_flow(service, "SES-EVAL4")
    booked = _appointments(db_path)[0]
    from fastapi.testclient import TestClient

    from .api import app

    with TestClient(app) as client:
        patient = client.post(
            "/patients",
            json={"full_name": "Alex Morgan", "dob": "1985-03-02", "phone": "612-555-0111"},
        ).json()
        response = client.post(
            "/appointments",
            json={
                "patient_ref": patient["patient_ref"],
                "npi": booked["npi"],
                "provider_name": booked["provider_name"],
                "slot": booked["slot"],
                "reason": "second opinion",
                "contact_method": "text",
                "idempotency_key": "eval-double-book",
                "confirmed": True,
            },
        )
    assert response.status_code == 409, f"double-book returned {response.status_code}: {response.text[:200]}"
    assert len(_appointments(db_path)) == 1, "a duplicate appointment was created"
    service.close()
    return f"second booking for {booked['slot']} refused with 409"


def scenario_expired_hold_refused() -> str:
    service, db_path = _fresh_service()
    from fastapi.testclient import TestClient

    from .api import app

    with TestClient(app) as client:
        providers = client.get("/providers?specialty=cardiology&city=Minneapolis&state=MN").json()
        npi = providers["providers"][0]["npi"]
        availability = client.get(f"/providers/{npi}/availability").json()
        slot = next(s["slot"] for s in availability["slots"] if s["available"])
        hold = client.post("/holds", json={"npi": npi, "slot": slot, "session_id": "SES-EVAL5"}).json()
        patient = client.post(
            "/patients",
            json={"full_name": "Alex Morgan", "dob": "1985-03-02", "phone": "612-555-0111"},
        ).json()
    # Expire the hold by moving its expiry into the past.
    conn = store.connect(db_path)
    try:
        conn.execute(
            "UPDATE holds SET expires_at = ? WHERE hold_id = ?",
            (time.time() - 60, hold["hold_id"]),
        )
        conn.commit()
    finally:
        conn.close()
    with TestClient(app) as client:
        response = client.post(
            "/appointments",
            json={
                "patient_ref": patient["patient_ref"],
                "npi": npi,
                "provider_name": providers["providers"][0]["name"],
                "slot": slot,
                "reason": "checkup",
                "contact_method": "text",
                "idempotency_key": "eval-expired-hold",
                "confirmed": True,
                "hold_id": hold["hold_id"],
            },
        )
    assert response.status_code in (409, 410), f"expired hold returned {response.status_code}: {response.text[:200]}"
    assert len(_appointments(db_path)) == 0, "an appointment was created on an expired hold"
    service.close()
    return f"booking on expired hold {hold['hold_id']} refused with {response.status_code}"


SCENARIOS = [
    ("find + book", scenario_find_and_book),
    ("book + reschedule", scenario_book_and_reschedule),
    ("correction mid-flow", scenario_correction_mid_flow),
    ("double-book refusal", scenario_double_book_refused),
    ("expired hold refusal", scenario_expired_hold_refused),
]


def main() -> int:
    print("Running deterministic evals (scripted model, no key needed)...")
    failures = 0
    for name, scenario in SCENARIOS:
        try:
            detail = scenario()
            print(f"  PASS  {name}: {detail}")
        except Exception as exc:  # noqa: BLE001 - eval reports, doesn't crash
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
