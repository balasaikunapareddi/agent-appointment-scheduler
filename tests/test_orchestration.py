"""Orchestration tests: the full multi-turn flow on the scripted model."""

from agent_appointment_scheduler import store
from agent_appointment_scheduler.chat import ChatService


def _run_booking(service, session_id):
    turns = [
        "I need a cardiologist near Minneapolis next week, I prefer mornings",
        "the second one",
        "Wednesday morning works",
        "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147",
        "Actually my phone is 612-555-0199",
        "It's a heart checkup follow-up, text me",
        "Yes, book it",
    ]
    replies = [service.chat(session_id, turn) for turn in turns]
    return replies


def test_full_booking_flow(db_path):
    service = ChatService(db_path=db_path, mock=True)
    replies = _run_booking(service, "SES-O1")
    last = replies[-1]
    assert "Booked!" in last.reply, last.reply
    assert last.booking_stage == "booked"
    assert "***-***-0199" in last.reply or "***-***-0199" in last.booking_summary
    conn = store.connect(db_path)
    try:
        appointments = conn.execute("SELECT * FROM appointments").fetchall()
    finally:
        conn.close()
    assert len(appointments) == 1
    assert appointments[0]["status"] == "booked"
    service.close()


def test_reschedule_in_same_session(db_path):
    service = ChatService(db_path=db_path, mock=True)
    _run_booking(service, "SES-O2")
    conn = store.connect(db_path)
    try:
        before = conn.execute("SELECT * FROM appointments").fetchone()
    finally:
        conn.close()
    reply = service.chat("SES-O2", "Actually, can we move it to Thursday morning instead?")
    assert "Moved it" in reply.reply, reply.reply
    conn = store.connect(db_path)
    try:
        after = conn.execute("SELECT * FROM appointments").fetchone()
    finally:
        conn.close()
    assert after["appointment_id"] == before["appointment_id"]
    assert after["slot"] != before["slot"]
    service.close()


def test_resume_in_fresh_service(db_path):
    service = ChatService(db_path=db_path, mock=True)
    _run_booking(service, "SES-O3")
    service.close()
    resumed = ChatService(db_path=db_path, mock=True)
    reply = resumed.chat("SES-O3", "What's my appointment?")
    assert "APT-" in reply.reply, reply.reply
    assert "cancel" not in reply.reply.lower() or "appointment" in reply.reply.lower()
    resumed.close()


def test_cancel_from_history(db_path):
    service = ChatService(db_path=db_path, mock=True)
    _run_booking(service, "SES-O4")
    reply = service.chat("SES-O4", "Cancel my appointment")
    assert "Cancelled" in reply.reply, reply.reply
    conn = store.connect(db_path)
    try:
        appointment = conn.execute("SELECT * FROM appointments").fetchone()
    finally:
        conn.close()
    assert appointment["status"] == "cancelled"
    service.close()


def test_tools_used_are_recorded(db_path):
    service = ChatService(db_path=db_path, mock=True)
    replies = _run_booking(service, "SES-O5")
    tools = [tool for reply in replies for tool in reply.tools_used]
    for expected in ("search_providers", "choose_provider", "get_availability", "choose_slot", "create_patient_record", "book_appointment"):
        assert expected in tools, f"{expected} not used; got {tools}"
    service.close()
