"""Shared fixtures: temp DB, ASGI transport, mock CMS."""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

os.environ["SCHEDULER_API_TRANSPORT"] = "asgi"


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / "test.db")
    os.environ["SCHEDULER_DB_PATH"] = path
    from agent_appointment_scheduler import store

    store.connect(path).close()
    return path


@pytest.fixture()
def client(db_path):
    from agent_appointment_scheduler.api import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def patient(client):
    response = client.post(
        "/patients",
        json={
            "full_name": "Jordan Lee",
            "dob": "1990-05-14",
            "phone": "612-555-0147",
            "member_id": "MBR-1234",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture()
def provider_and_slot(client):
    providers = client.get("/providers?specialty=cardiology&city=Minneapolis&state=MN").json()
    assert providers["providers"], "no providers returned"
    npi = providers["providers"][0]["npi"]
    availability = client.get(f"/providers/{npi}/availability").json()
    slot = next(s["slot"] for s in availability["slots"] if s["available"])
    return providers["providers"][0], slot


@pytest.fixture()
def booked(client, patient, provider_and_slot):
    provider, slot = provider_and_slot
    response = client.post(
        "/appointments",
        json={
            "patient_ref": patient["patient_ref"],
            "npi": provider["npi"],
            "provider_name": provider["name"],
            "slot": slot,
            "reason": "heart checkup follow-up",
            "contact_method": "text",
            "idempotency_key": "test-booked-1",
            "confirmed": True,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()
