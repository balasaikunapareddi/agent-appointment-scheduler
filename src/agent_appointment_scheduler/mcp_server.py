"""MCP server exposing the scheduling tools.

Run it directly (stdio transport, for local dev and for the Strands agent)::

    python -m agent_appointment_scheduler.mcp_server

Any MCP client can use the same server. All tools call the FastAPI
scheduling service over HTTP (``SCHEDULER_API_URL``, default
``http://127.0.0.1:8000``); for tests, set ``SCHEDULER_API_TRANSPORT=asgi``
to call the app in-process with no port. All deciding happens in the
service and in the deterministic modules (``slots``, ``policy``,
``planner``); these tool functions are the protocol-facing wrappers.

PHI rule: tools only ever carry a patient reference (PAT-...) and masked
values. Sensitive details are stored encrypted by the service and never
appear in tool arguments or results.
"""

from __future__ import annotations

import json
import logging
import os
import time

import httpx
from mcp.server.mcpserver import MCPServer

from . import slots as slot_logic
from .phi import PIIRedactor
from .planner import plan_for
from .sessions import SessionStore

logger = logging.getLogger(__name__)

API_URL = os.environ.get("SCHEDULER_API_URL", "http://127.0.0.1:8000")
DB_PATH = os.environ.get("SCHEDULER_DB_PATH", "scheduler.db")
TOOL_TIMEOUT_SECONDS = 15.0
TOOL_MAX_RETRIES = 2

server = MCPServer(
    name="appointment-scheduler",
    description="Deterministic appointment scheduling tools (CMS providers, FastAPI booking).",
    instructions=(
        "Tools for finding providers, checking availability, holding slots, "
        "and booking, rescheduling, or cancelling appointments. All data is "
        "synthetic mock data except provider directory listings from the "
        "public CMS NPI Registry. Booking is idempotent and requires explicit "
        "user confirmation."
    ),
)


def _client() -> httpx.Client:
    """HTTP client to the scheduling service (real HTTP, or in-process ASGI).

    Note: httpx's ASGITransport is async-only, so the in-process path
    uses starlette's TestClient, which is a synchronous httpx client
    over the same app. Only used when SCHEDULER_API_TRANSPORT=asgi
    (tests and the eval suite); production always uses real HTTP.
    """
    if os.environ.get("SCHEDULER_API_TRANSPORT") == "asgi":
        from fastapi.testclient import TestClient

        from .api import app

        return TestClient(app, base_url="http://test")
    return httpx.Client(base_url=API_URL, timeout=TOOL_TIMEOUT_SECONDS, trust_env=False)
    # trust_env=False: the service is always on localhost, so proxies
    # never apply — and it avoids httpx failing to parse a malformed
    # no_proxy list in some sandboxes.


def _call(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the service with bounded retries and backoff."""
    last_error: Exception | None = None
    for attempt in range(TOOL_MAX_RETRIES + 1):
        try:
            with _client() as client:
                response = client.request(method, path, json=payload)
            if response.status_code >= 400:
                detail = response.json().get("detail", response.text) if response.text else response.reason_phrase
                return {"ok": False, "error": str(detail), "status": response.status_code}
            return {"ok": True, **_as_dict(response.json())}
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("service call failed (attempt %d): %s", attempt + 1, exc)
            if attempt < TOOL_MAX_RETRIES:
                time.sleep(0.5 * (attempt + 1))
    return {"ok": False, "error": f"Service unreachable: {last_error}"}


def _as_dict(data) -> dict:
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        return {"items": data}
    return {"value": data}


def _store() -> SessionStore:
    # Read the path at call time: tests point SCHEDULER_DB_PATH at a
    # temp file per test, and the module may have been imported earlier.
    return SessionStore(os.environ.get("SCHEDULER_DB_PATH", DB_PATH))


def _form(session_id: str):
    store = _store()
    return store, store.get_booking_form(session_id)


@server.tool()
def search_providers(specialty: str, city: str = "Minneapolis", state: str = "MN", session_id: str = "") -> dict:
    """Find providers by specialty and city, from the CMS NPI Registry.

    Pass a specialty name like 'cardiology' or 'family medicine' — the
    service maps it to the CMS taxonomy description. Returns providers with
    name, specialty, city, and phone, numbered for follow-ups.
    """
    result = _call(
        "GET",
        f"/providers?specialty={specialty}&city={city}&state={state}&limit=10",
    )
    if not result.get("ok"):
        return result
    providers = result.get("providers", [])
    if session_id:
        store, form = _form(session_id)
        form.last_providers = providers
        if providers:
            form.stage = "providers_shown"
        store.save_booking_form(session_id, form)
    return {
        "ok": True,
        "providers": providers,
        "source": result.get("source"),
        "note": result.get("note", ""),
    }


@server.tool()
def get_availability(npi: str, date_from: str = "", date_to: str = "", time_of_day: str = "") -> dict:
    """Show open appointment slots for a provider over a date range.

    Dates are YYYY-MM-DD. time_of_day is 'morning' or 'afternoon' (or
    empty for all day). Only free slots are returned as available.
    """
    query = f"/providers/{npi}/availability?date_from={date_from}&date_to={date_to}"
    if time_of_day:
        query += f"&time_of_day={time_of_day}"
    result = _call("GET", query)
    if not result.get("ok"):
        return result
    return {"ok": True, "npi": npi, "slots": result.get("slots", [])}


@server.tool()
def create_patient_record(
    full_name: str, dob: str, phone: str, member_id: str = "", contact_method: str = "text", session_id: str = ""
) -> dict:
    """Store patient details securely and return a reference.

    Sensitive values are encrypted at rest by the service. Only the
    patient reference (PAT-...) and masked values come back — the agent
    never sees or repeats the raw values.

    Name, date of birth and phone are validated here, before anything
    is stored, so a bad value comes back as a plain error the agent can
    repeat to the user and ask again — never as a half-built record and
    never as a reason to go looking for someone else's record.
    """
    name, err = slot_logic.validate_name(full_name)
    if err:
        return {"ok": False, "error": err}
    dob_norm, err = slot_logic.validate_dob(dob)
    if err:
        return {"ok": False, "error": err}
    phone_norm, err = slot_logic.validate_phone(phone)
    if err:
        return {"ok": False, "error": err}
    result = _call(
        "POST",
        "/patients",
        {
            "full_name": name,
            "dob": dob_norm,
            "phone": phone_norm,
            "member_id": member_id,
            "contact_method": contact_method,
        },
    )
    if not result.get("ok"):
        return result
    if session_id:
        store, form = _form(session_id)
        form.patient_ref = result["patient_ref"]
        form.patient_masked = result["name_masked"]
        form.phone_masked = result["phone_masked"]
        form.contact_method = result.get("contact_method") or form.contact_method
        store.save_booking_form(session_id, form)
        # Persist the literals and re-redact the session's stored history
        # (the user message that gave the name was stored before this
        # record existed). Register the normalized values — those are
        # the ones the service stored.
        store.register_session_sensitive_values(session_id, name, dob_norm, phone_norm, member_id)
    return {
        "ok": True,
        "patient_ref": result["patient_ref"],
        "name_masked": result["name_masked"],
        "dob_masked": result["dob_masked"],
        "phone_masked": result["phone_masked"],
        "contact_method": result.get("contact_method"),
    }


@server.tool()
def update_patient_contact(patient_ref: str, phone: str, session_id: str = "") -> dict:
    """Update the phone number on a patient record (e.g. a correction).

    The new number is validated first: a correction that is not a real
    10-digit number is rejected with a plain error instead of being
    stored.
    """
    phone_norm, err = slot_logic.validate_phone(phone)
    if err:
        return {"ok": False, "error": err}
    result = _call("PATCH", f"/patients/{patient_ref}", {"phone": phone_norm})
    if not result.get("ok"):
        return result
    if session_id:
        store, form = _form(session_id)
        if form.patient_ref == patient_ref:
            form.phone_masked = result["phone_masked"]
            store.save_booking_form(session_id, form)
    return {
        "ok": True,
        "patient_ref": patient_ref,
        "phone_masked": result["phone_masked"],
    }


@server.tool()
def choose_provider(choice: str, session_id: str) -> dict:
    """Pick a provider from the last search: 'second one', '#2', or a name."""
    store, form = _form(session_id)
    provider = slot_logic.parse_provider_choice(choice, form.last_providers)
    if provider is None:
        return {
            "ok": False,
            "error": "I couldn't tell which provider you meant. Say 'the first one' or 'the second one', or give the name.",
        }
    form.provider_npi = provider["npi"]
    form.provider_name = provider["name"]
    form.stage = "providers_shown"
    store.save_booking_form(session_id, form)
    return {"ok": True, "provider": provider}


@server.tool()
def choose_slot(slot: str, session_id: str) -> dict:
    """Pick a slot (YYYY-MM-DDTHH:MM) and hold it for 10 minutes.

    The slot is re-checked against live availability first; a hold keeps
    it reserved while the user confirms the booking.
    """
    store, form = _form(session_id)
    if not form.provider_npi:
        return {"ok": False, "error": "Pick a provider first."}
    avail = _call(
        "GET",
        f"/providers/{form.provider_npi}/availability?date_from={slot[:10]}&date_to={slot[:10]}",
    )
    if not avail.get("ok"):
        return avail
    wanted = next((s for s in avail.get("slots", []) if s["slot"] == slot), None)
    if wanted is None or not wanted.get("available"):
        return {"ok": False, "error": f"Slot {slot} is not available. Pick another time."}
    hold = _call(
        "POST", "/holds", {"npi": form.provider_npi, "slot": slot, "session_id": session_id}
    )
    if not hold.get("ok"):
        return hold
    form.slot = slot
    form.date, _, form.time = slot.partition("T")
    form.hold_id = hold["hold_id"]
    form.stage = "slot_picked"
    store.save_booking_form(session_id, form)
    return {"ok": True, "slot": slot, "hold_id": hold["hold_id"], "expires_in_seconds": hold["expires_in_seconds"]}


@server.tool()
def set_visit_details(reason: str = "", contact_method: str = "", session_id: str = "") -> dict:
    """Set the reason for the visit and how the clinic should reach the patient."""
    store, form = _form(session_id)
    if reason:
        form.reason = reason.strip()[:500]
    if contact_method:
        normalized = contact_method.strip().lower()
        if normalized not in ("text", "call"):
            return {"ok": False, "error": "Contact method must be 'text' or 'call'."}
        form.contact_method = normalized
        store.set_preference(session_id, "contact_method", normalized)
    store.save_booking_form(session_id, form)
    return {"ok": True, "reason": form.reason, "contact_method": form.contact_method}


@server.tool()
def get_booking_status(session_id: str) -> dict:
    """Show the booking form: what is filled, what is missing, the masked
    summary, and the next question. Use before asking the user to confirm."""
    store, form = _form(session_id)
    summary = form.summary()
    return {
        "ok": True,
        "stage": form.stage,
        "missing": summary.missing,
        "summary": summary.model_dump(),
        "readable_summary": summary.readable(),
        "next_question": form.next_question(),
    }


@server.tool()
def get_plan(session_id: str) -> dict:
    """Get the explicit plan for the current booking stage.

    Plan-then-act: follow these steps in order. If a step fails, call
    get_plan again and follow the replanned steps from the failed one.
    """
    store, form = _form(session_id)
    plan = plan_for(form.stage)
    return {"ok": True, "plan": plan.model_dump(), "readable": plan.readable()}


@server.tool()
def book_appointment(session_id: str, confirmed: bool = False, idempotency_key: str = "") -> dict:
    """Book the appointment in the current form.

    Requires every slot filled, a live availability re-check, explicit
    user confirmation (confirmed=true), and an idempotency key. The same
    key twice returns the original appointment — never a duplicate.
    """
    store, form = _form(session_id)
    missing = form.missing_slots()
    if missing:
        return {"ok": False, "error": f"Still missing: {', '.join(missing)}. Ask for the next piece first."}
    key = idempotency_key.strip() or f"{session_id}-{form.slot}"
    result = _call(
        "POST",
        "/appointments",
        {
            "patient_ref": form.patient_ref,
            "npi": form.provider_npi,
            "provider_name": form.provider_name,
            "slot": form.slot,
            "reason": form.reason,
            "contact_method": form.contact_method or "text",
            "idempotency_key": key,
            "confirmed": confirmed,
            "hold_id": form.hold_id or None,
        },
    )
    if not result.get("ok"):
        return result
    form.appointment_id = result["appointment_id"]
    form.stage = "booked"
    store.save_booking_form(session_id, form)
    return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}


@server.tool()
def list_appointments(session_id: str = "", patient_ref: str = "") -> dict:
    """List appointments for this session's patient (or a given reference).

    Scoped by design: without a patient reference — the one this
    session collected, or one passed in explicitly — this returns an
    empty list, never the whole appointment book. A session must never
    see another patient's appointments.
    """
    ref = patient_ref
    if not ref and session_id:
        store, form = _form(session_id)
        ref = form.patient_ref
    if not ref:
        return {
            "ok": True,
            "appointments": [],
            "note": "No patient on file for this session yet — collect the patient's details first.",
        }
    result = _call("GET", f"/appointments?patient_ref={ref}")
    if not result.get("ok"):
        return result
    return {"ok": True, "appointments": result.get("items", [])}


@server.tool()
def cancel_appointment(appointment_id: str, session_id: str = "") -> dict:
    """Cancel an appointment by id. Refuses an already-cancelled one."""
    result = _call("POST", f"/appointments/{appointment_id}/cancel", {})
    if not result.get("ok"):
        return result
    if session_id:
        store, form = _form(session_id)
        if form.appointment_id == appointment_id:
            form.appointment_id = ""
            form.stage = "collecting"
            store.save_booking_form(session_id, form)
    return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}


@server.tool()
def reschedule_appointment(appointment_id: str, new_slot: str, session_id: str = "") -> dict:
    """Move an appointment to a new slot (YYYY-MM-DDTHH:MM).

    The new slot is re-checked for availability first. The old slot is
    freed and the appointment id stays the same.
    """
    result = _call(
        "POST", f"/appointments/{appointment_id}/reschedule", {"new_slot": new_slot}
    )
    if not result.get("ok"):
        return result
    if session_id:
        store, form = _form(session_id)
        if form.appointment_id == appointment_id:
            form.slot = new_slot
            form.date, _, form.time = new_slot.partition("T")
            store.save_booking_form(session_id, form)
    return {"ok": True, **{k: v for k, v in result.items() if k != "ok"}}


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
