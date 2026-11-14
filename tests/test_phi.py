"""PHI tests: the LLM context never contains a raw sensitive value.

This is the proof for the pass-by-reference pattern: a booking turn
runs end to end and the agent's message history / tool arguments are
asserted to contain only the patient reference and masked forms.
"""

import json

from agent_appointment_scheduler.chat import ChatService
from agent_appointment_scheduler.crypto import decrypt_field, encrypt_field
from agent_appointment_scheduler.phi import PIIRedactor, mask_dob, mask_member_id, mask_name, mask_phone


def test_masking():
    assert mask_phone("612-555-0147") == "***-***-0147"
    assert mask_phone("(612) 555-0147") == "***-***-0147"
    assert mask_dob("1990-05-14") == "**/**/1990"
    assert mask_member_id("MBR-1234") == "***-***-1234"
    assert mask_name("Jordan Lee") == "J*** L***"


def test_redactor_masks_literals_and_patterns():
    redactor = PIIRedactor()
    redactor.register("Jordan Lee", "612-555-0147", "1990-05-14", "MBR-1234")
    text = redactor.redact(
        "Jordan Lee (MBR-1234) was born 1990-05-14, phone 612-555-0147, "
        "also reachable at 651-555-0199, DOB 05/14/1990."
    )
    assert "Jordan Lee" not in text
    assert "612-555-0147" not in text
    assert "1990-05-14" not in text
    assert "MBR-1234" not in text
    assert "651-555-0199" not in text
    assert "05/14/1990" not in text


def test_field_encryption_round_trip(db_path, monkeypatch):
    from cryptography.fernet import Fernet

    monkeypatch.setenv("SCHEDULER_ENCRYPTION_KEY", Fernet.generate_key().decode())
    ciphertext = encrypt_field("612-555-0147")
    assert ciphertext != "612-555-0147"
    assert decrypt_field(ciphertext) == "612-555-0147"


def test_patient_record_encrypted_at_rest(client, db_path):
    response = client.post(
        "/patients",
        json={"full_name": "Jordan Lee", "dob": "1990-05-14", "phone": "612-555-0147", "member_id": "MBR-1234"},
    )
    assert response.status_code == 201
    from agent_appointment_scheduler import store

    conn = store.connect(db_path)
    try:
        row = conn.execute("SELECT * FROM patients").fetchone()
    finally:
        conn.close()
    assert row is not None
    raw = json.dumps(dict(row))
    assert "Jordan Lee" not in raw
    assert "612-555-0147" not in raw
    assert "1990-05-14" not in raw
    assert "MBR-1234" not in raw


def test_llm_context_never_contains_raw_pii(db_path):
    """The core proof: run a booking turn and assert the agent's message
    history and tool arguments hold only the reference and masked forms."""
    service = ChatService(db_path=db_path, mock=True)
    session_id = "SES-PHI1"
    turns = [
        "I need a cardiologist near Minneapolis next week, I prefer mornings",
        "the second one",
        "Wednesday morning works",
        "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147",
        "It's a heart checkup follow-up, text me",
        "Yes, book it",
    ]
    for turn in turns:
        service.chat(session_id, turn)
    service.close()
    from agent_appointment_scheduler import store

    conn = store.connect(db_path)
    try:
        rows = conn.execute("SELECT content FROM session_messages WHERE session_id = ?", (session_id,)).fetchall()
        traces = conn.execute("SELECT payload FROM traces WHERE session_id = ?", (session_id,)).fetchall()
    finally:
        conn.close()
    forbidden = ["Jordan Lee", "612-555-0147", "1990-05-14", "05/14/1990"]
    for row in rows:
        for value in forbidden:
            assert value not in row["content"], f"raw PII {value!r} in stored history: {row['content'][:120]}"
    for trace in traces:
        for value in forbidden:
            assert value not in trace["payload"], f"raw PII {value!r} in trace"
    # And the masked forms ARE there — the agent still knows who it booked.
    joined = " ".join(r["content"] for r in rows)
    assert "PAT-" in joined
    assert "***-***-0147" in joined
