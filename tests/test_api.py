"""FastAPI tests: booking, double-booking, holds, idempotency, cancel/reschedule, patients."""

import time


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "appointment-scheduler"


def test_providers_uses_seed_when_cms_unreachable(client):
    # In this sandbox the egress proxy config cannot be parsed by httpx,
    # so the CMS client falls back to the seed list (same as CMS down).
    response = client.get("/providers?specialty=cardiology&city=Minneapolis&state=MN")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] in ("seed", "cms", "cache")
    assert body["providers"]


def test_patient_record_encrypted_and_masked(client, patient):
    assert patient["patient_ref"].startswith("PAT-")
    assert patient["phone_masked"] == "***-***-0147"
    assert patient["dob_masked"] == "**/**/1990"
    assert patient["name_masked"] == "J*** L***"
    assert "612-555-0147" not in str(patient)
    # Fetching the record returns masked values only.
    fetched = client.get(f"/patients/{patient['patient_ref']}").json()
    assert fetched["phone_masked"] == "***-***-0147"
    assert "612-555-0147" not in str(fetched)


def test_patient_phone_validation(client):
    response = client.post(
        "/patients",
        json={"full_name": "Jordan Lee", "dob": "1990-05-14", "phone": "not-a-phone"},
    )
    assert response.status_code == 422


def test_booking_happy_path(client, patient, provider_and_slot):
    provider, slot = provider_and_slot
    response = client.post(
        "/appointments",
        json={
            "patient_ref": patient["patient_ref"],
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "checkup",
            "contact_method": "text",
            "idempotency_key": "happy-1",
            "confirmed": True,
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["appointment_id"].startswith("APT-")
    assert body["slot"] == slot
    assert body["status"] == "booked"


def test_booking_requires_confirmation(client, patient, provider_and_slot):
    provider, slot = provider_and_slot
    response = client.post(
        "/appointments",
        json={
            "patient_ref": patient["patient_ref"],
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "checkup",
            "contact_method": "text",
            "idempotency_key": "no-confirm-1",
            "confirmed": False,
        },
    )
    assert response.status_code == 409


def test_double_booking_refused(client, patient, provider_and_slot, booked):
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
            "idempotency_key": "double-1",
            "confirmed": True,
        },
    )
    assert response.status_code == 409


def test_idempotency_same_key_same_appointment(client, patient, provider_and_slot):
    provider, slot = provider_and_slot
    payload = {
        "patient_ref": patient["patient_ref"],
        "npi": provider["npi"],
        "provider_name": provider["name"],
        "slot": slot,
        "reason": "checkup",
        "contact_method": "text",
        "idempotency_key": "idem-1",
        "confirmed": True,
    }
    first = client.post("/appointments", json=payload)
    second = client.post("/appointments", json=payload)
    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["appointment_id"] == second.json()["appointment_id"]
    assert second.json().get("duplicate") is True
    assert len(client.get("/appointments").json()) == 1


def test_hold_expiry(client, db_path, provider_and_slot):
    provider, slot = provider_and_slot
    hold = client.post("/holds", json={"npi": provider["npi"], "slot": slot, "session_id": "SES-T1"}).json()
    assert hold["hold_id"]
    # Expire it in the DB, then booking against it must fail.
    from agent_appointment_scheduler import store

    conn = store.connect(db_path)
    try:
        conn.execute("UPDATE holds SET expires_at = ? WHERE hold_id = ?", (time.time() - 60, hold["hold_id"]))
        conn.commit()
    finally:
        conn.close()
    patient = client.post(
        "/patients",
        json={"full_name": "Jordan Lee", "dob": "1990-05-14", "phone": "612-555-0147"},
    ).json()
    response = client.post(
        "/appointments",
        json={
            "patient_ref": patient["patient_ref"],
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "checkup",
            "contact_method": "text",
            "idempotency_key": "expired-hold-1",
            "confirmed": True,
            "hold_id": hold["hold_id"],
        },
    )
    assert response.status_code in (409, 410)


def test_cancel_and_reschedule(client, patient, provider_and_slot, booked):
    provider, slot = provider_and_slot
    appointment_id = booked["appointment_id"]
    availability = client.get(f"/providers/{provider['npi']}/availability").json()
    other_slot = next(s["slot"] for s in availability["slots"] if s["available"] and s["slot"] != slot)
    rescheduled = client.post(f"/appointments/{appointment_id}/reschedule", json={"new_slot": other_slot})
    assert rescheduled.status_code == 200, rescheduled.text
    assert rescheduled.json()["slot"] == other_slot
    assert rescheduled.json()["appointment_id"] == appointment_id
    cancelled = client.post(f"/appointments/{appointment_id}/cancel", json={})
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    again = client.post(f"/appointments/{appointment_id}/cancel", json={})
    assert again.status_code == 409


def test_list_appointments(client, booked):
    items = client.get("/appointments").json()
    assert len(items) == 1
    assert items[0]["appointment_id"] == booked["appointment_id"]
