"""Guards added after the live robustness test (2026-11-14, 14/16 PASS).

Two failures, two fixes, tested here:

1. Double-booking across sessions: the in-process guards (availability
   excludes booked slots; ``check_book`` 409s) were correct, but the
   local SQLite file was wiped by every Render free-tier spin-down, so
   the guards had nothing to check against. The durable fix is the
   hosted database (``SCHEDULER_DATABASE_URL`` -> libsql/Turso); these
   tests pin the guard behaviour itself.
2. Invalid contact info: a garbage phone sailed past the form and the
   agent presented a different patient's record. Contact details are
   now validated deterministically before anything is stored, and
   ``list_appointments`` never returns another patient's appointments.
"""

import os

import pytest

from agent_appointment_scheduler import mcp_server, store


def _book(client, patient_ref, provider, slot, key):
    response = client.post(
        "/appointments",
        json={
            "patient_ref": patient_ref,
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "heart checkup follow-up",
            "contact_method": "text",
            "idempotency_key": key,
            "confirmed": True,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_availability_excludes_booked_slot(client, patient, provider_and_slot, booked):
    provider, slot = provider_and_slot
    date = slot[:10]
    availability = client.get(
        f"/providers/{provider['npi']}/availability?date_from={date}&date_to={date}"
    ).json()
    wanted = next(s for s in availability["slots"] if s["slot"] == slot)
    assert wanted["available"] is False, "a booked slot must not be listed as available"


def test_second_booking_same_slot_refused(client, patient, provider_and_slot, booked):
    provider, slot = provider_and_slot
    other = client.post(
        "/patients",
        json={"full_name": "Alex Morgan", "dob": "1985-03-02", "phone": "612-555-0111"},
    ).json()
    response = client.post(
        "/appointments",
        json={
            "patient_ref": other["patient_ref"],
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "second opinion",
            "contact_method": "text",
            "idempotency_key": "test-second-session",
            "confirmed": True,
        },
    )
    assert response.status_code == 409, response.text


def test_patient_dob_validation(client):
    for bad_dob in ("tomorrow", "next Tuesday", "not a date", "2099-01-01"):
        response = client.post(
            "/patients",
            json={"full_name": "Jordan Lee", "dob": bad_dob, "phone": "612-555-0147"},
        )
        assert response.status_code == 422, f"dob {bad_dob!r} returned {response.status_code}"
    for good_dob in ("1990-05-14", "05/14/1990", "May 14, 1990"):
        response = client.post(
            "/patients",
            json={"full_name": "Jordan Lee", "dob": good_dob, "phone": "612-555-0147"},
        )
        assert response.status_code == 201, f"dob {good_dob!r}: {response.text}"


def test_create_patient_tool_rejects_bad_contact(db_path):
    result = mcp_server.create_patient_record(
        "Jordan Lee", "1990-05-14", "abc", session_id="SES-G1"
    )
    assert result["ok"] is False
    assert "10 digits" in result["error"]
    result = mcp_server.create_patient_record(
        "Jordan Lee", "tomorrow", "612-555-0147", session_id="SES-G1"
    )
    assert result["ok"] is False
    result = mcp_server.create_patient_record(
        "1", "1990-05-14", "612-555-0147", session_id="SES-G1"
    )
    assert result["ok"] is False
    conn = store.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    conn.close()
    assert count == 0, "no patient record may be created from rejected details"


def test_update_patient_contact_tool_rejects_bad_phone(db_path):
    created = mcp_server.create_patient_record(
        "Jordan Lee", "1990-05-14", "612-555-0147", session_id="SES-G1"
    )
    assert created["ok"] is True
    result = mcp_server.update_patient_contact(created["patient_ref"], "abc", session_id="SES-G1")
    assert result["ok"] is False
    assert "10 digits" in result["error"]


def test_list_appointments_is_session_scoped(client, db_path):
    providers = client.get("/providers?specialty=cardiology&city=Minneapolis&state=MN").json()
    provider = providers["providers"][0]
    slots = [
        s["slot"]
        for s in client.get(f"/providers/{provider['npi']}/availability").json()["slots"]
        if s["available"]
    ][:2]
    a = mcp_server.create_patient_record(
        "Jordan Lee", "1990-05-14", "612-555-0147", session_id="SES-A"
    )
    b = mcp_server.create_patient_record(
        "Alex Morgan", "1985-03-02", "612-555-0111", session_id="SES-B"
    )
    assert a["ok"] is True and b["ok"] is True
    _book(client, a["patient_ref"], provider, slots[0], "guard-a")
    _book(client, b["patient_ref"], provider, slots[1], "guard-b")

    fresh = mcp_server.list_appointments(session_id="SES-FRESH")
    assert fresh["ok"] is True
    assert fresh["appointments"] == [], "a session with no patient sees nothing"

    no_args = mcp_server.list_appointments()
    assert no_args["ok"] is True
    assert no_args["appointments"] == [], "no reference at all sees nothing"

    only_a = mcp_server.list_appointments(session_id="SES-A")
    assert [x["patient_ref"] for x in only_a["appointments"]] == [a["patient_ref"]]
    only_b = mcp_server.list_appointments(patient_ref=b["patient_ref"])
    assert [x["patient_ref"] for x in only_b["appointments"]] == [b["patient_ref"]]


def test_libsql_adapter_row_semantics(tmp_path):
    import libsql

    path = str(tmp_path / "adapter.db")
    conn = store._LibsqlConnection(libsql.connect(path))
    conn.executescript(store.SCHEMA)
    conn.commit()
    conn.execute(
        "INSERT INTO patients (patient_ref, full_name_enc, dob_enc, phone_enc, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ["PAT-AAA", "x", "y", "z", 1.0],
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_ref = ?", ["PAT-AAA"]
    ).fetchone()
    assert row["patient_ref"] == "PAT-AAA"
    assert row[0] == "PAT-AAA"
    assert row.get("missing", "dflt") == "dflt"
    assert dict(row)["phone_enc"] == "z"
    assert len(row) == 7
    assert list(row)[0] == "PAT-AAA"
    with pytest.raises(IndexError):
        row["no_such_column"]
    assert conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0] == 1
    conn.close()


def test_connect_routes_to_remote_url(tmp_path, monkeypatch):
    # A file: URL exercises the libsql path end to end with no network.
    remote_path = str(tmp_path / "remote.db")
    monkeypatch.setenv("SCHEDULER_DATABASE_URL", f"file:{remote_path}")
    monkeypatch.setenv("SCHEDULER_DATABASE_AUTH_TOKEN", "")
    conn = store.connect("ignored-local.db")
    conn.execute(
        "INSERT INTO patients (patient_ref, full_name_enc, dob_enc, phone_enc, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ["PAT-REMOTE", "x", "y", "z", 1.0],
    )
    conn.commit()
    row = conn.execute(
        "SELECT patient_ref FROM patients WHERE patient_ref = ?", ["PAT-REMOTE"]
    ).fetchone()
    assert row["patient_ref"] == "PAT-REMOTE"
    conn.close()
    # The local file was never created: the remote path was used.
    assert not os.path.exists("ignored-local.db")
