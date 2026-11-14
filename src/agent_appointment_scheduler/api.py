"""FastAPI scheduling service — the system of record.

Everything the agent does goes through here over HTTP. Deterministic
rules live in this service (and in ``policy``), not in the model:
no double-booking, slot holds with expiry, idempotent booking, Pydantic
validation on every input, and structured logging with PII redaction.

Endpoints:
- ``GET /health`` — liveness.
- ``GET /providers`` — CMS NPI Registry search (cached, seed fallback).
- ``GET /providers/{npi}/availability`` — deterministic slots.
- ``POST /holds`` — hold a slot for 10 minutes while the user confirms.
- ``POST /patients`` / ``GET /patients/{ref}`` / ``PATCH /patients/{ref}`` /
  ``DELETE /patients/{ref}`` — patient records; sensitive fields are
  Fernet-encrypted at rest and only masked values ever leave this service.
- ``POST /appointments`` — book (confirmation + idempotency key required).
- ``GET /appointments`` — list by patient reference.
- ``POST /appointments/{id}/cancel`` and ``/reschedule``.
- ``GET /sessions`` / ``GET /sessions/{id}`` — session list + state for the UI.
- ``POST /chat`` — talk to the agent (session id + message in, reply +
  booking state out). Served by the same app as the UI.
- ``GET /`` and ``GET /app.js`` — the single-page chat UI (static/).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import crypto, policy, slots, store
from .cms import CMSClient, DEFAULT_CITY, DEFAULT_STATE, taxonomy_code_for
from .phi import PIIRedactor, mask_member_id, mask_name, mask_phone, mask_dob

logger = logging.getLogger(__name__)

MORNING_TIMES = ["08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30"]
AFTERNOON_TIMES = ["13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00", "16:30"]
HOLD_TTL_SECONDS = policy.HOLD_TTL_SECONDS


def db_path() -> str:
    return os.environ.get("SCHEDULER_DB_PATH", "scheduler.db")


def get_conn() -> sqlite3.Connection:
    conn = store.connect(db_path())
    try:
        yield conn
    finally:
        conn.close()


def _redact_filter() -> logging.Filter:
    redactor = PIIRedactor()

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            if isinstance(record.msg, str):
                record.msg = redactor.redact(record.msg)
            return True

    return _Filter()


logging.getLogger("agent_appointment_scheduler").addFilter(_redact_filter())

app = FastAPI(title="Appointment Scheduler", version="0.1.0")


# ---------------------------------------------------------------------------
# Schemas (structured outputs — Pydantic everywhere, no free-text parsing)
# ---------------------------------------------------------------------------


class ProviderOut(BaseModel):
    npi: str
    name: str
    credential: str = ""
    specialty: str = ""
    taxonomy_code: str = ""
    organization: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    phone: str = ""
    seed: bool = False


class ProvidersResponse(BaseModel):
    providers: list[ProviderOut]
    source: str
    note: str = ""


class SlotOut(BaseModel):
    slot: str  # "2026-11-18T09:00"
    date: str
    time: str
    time_of_day: str
    available: bool = True


class AvailabilityResponse(BaseModel):
    npi: str
    slots: list[SlotOut]


class HoldRequest(BaseModel):
    npi: str
    slot: str
    session_id: str


class HoldOut(BaseModel):
    hold_id: str
    npi: str
    slot: str
    expires_at: float
    expires_in_seconds: int


class PatientCreate(BaseModel):
    full_name: str = Field(min_length=2, max_length=100)
    dob: str = Field(description="YYYY-MM-DD or MM/DD/YYYY")
    phone: str
    member_id: str = ""
    contact_method: str = "text"


class PatientOut(BaseModel):
    patient_ref: str
    name_masked: str
    dob_masked: str
    phone_masked: str
    member_id_masked: str | None = None
    contact_method: str | None = None


class PatientUpdate(BaseModel):
    phone: str | None = None
    contact_method: str | None = None


class AppointmentCreate(BaseModel):
    patient_ref: str
    npi: str
    provider_name: str = ""
    slot: str
    reason: str = ""
    contact_method: str = "text"
    idempotency_key: str
    confirmed: bool = False
    hold_id: str | None = None


class AppointmentOut(BaseModel):
    appointment_id: str
    patient_ref: str
    npi: str
    provider_name: str
    slot: str
    date: str
    time: str
    reason: str = ""
    contact_method: str = ""
    status: str
    duplicate: bool = False


class RescheduleRequest(BaseModel):
    new_slot: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    booking_stage: str = ""
    booking_summary: str = ""
    missing_slots: list[str] = []
    preferences: dict = {}
    skills_activated: list[str] = []
    cache: dict = {}
    tools_used: list[str] = []


# ---------------------------------------------------------------------------
# Slots: deterministic generation
# ---------------------------------------------------------------------------


def _slot_available(npi: str, slot: str) -> bool:
    """Deterministic pseudo-availability: ~75% of slots are open.

    Same inputs always give the same answer, so demos and tests are stable.
    Booked appointments and active holds remove slots on top of this.
    """
    digest = hashlib.sha256(f"{npi}|{slot}".encode()).hexdigest()
    return int(digest[:8], 16) % 4 != 0


def generate_slots(
    conn: sqlite3.Connection,
    npi: str,
    date_from: date,
    date_to: date,
    time_of_day: str | None = None,
) -> list[SlotOut]:
    times: list[tuple[str, str]] = []
    if time_of_day in (None, "morning"):
        times += [(t, "morning") for t in MORNING_TIMES]
    if time_of_day in (None, "afternoon"):
        times += [(t, "afternoon") for t in AFTERNOON_TIMES]
    booked = {
        row[0]
        for row in conn.execute(
            "SELECT slot FROM appointments WHERE npi = ? AND status = 'booked'",
            (npi,),
        ).fetchall()
    }
    now = time.time()
    held = {
        row[0]
        for row in conn.execute(
            "SELECT slot FROM holds WHERE npi = ? AND expires_at > ?", (npi, now)
        ).fetchall()
    }
    out: list[SlotOut] = []
    day = date_from
    while day <= date_to:
        if day.weekday() < 5:  # clinics run Mon-Fri
            for clock, tod in times:
                slot = f"{day.isoformat()}T{clock}"
                available = (
                    _slot_available(npi, slot)
                    and slot not in booked
                    and slot not in held
                )
                out.append(
                    SlotOut(
                        slot=slot,
                        date=day.isoformat(),
                        time=clock,
                        time_of_day=tod,
                        available=available,
                    )
                )
        day += timedelta(days=1)
    return out


def _to_appointment_out(row: dict, duplicate: bool = False) -> AppointmentOut:
    slot = row["slot"]
    day, clock = slot.split("T") if "T" in slot else (slot, "")
    return AppointmentOut(
        appointment_id=row["appointment_id"],
        patient_ref=row["patient_ref"],
        npi=row["npi"],
        provider_name=row["provider_name"],
        slot=slot,
        date=day,
        time=clock,
        reason=row.get("reason") or "",
        contact_method=row.get("contact_method") or "",
        status=row["status"],
        duplicate=duplicate,
    )


def _patient_row(conn: sqlite3.Connection, patient_ref: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM patients WHERE patient_ref = ?", (patient_ref,)
    ).fetchone()
    if row is None:
        return None
    data = dict(row)
    return {
        "patient_ref": data["patient_ref"],
        "full_name": crypto.decrypt_field(data["full_name_enc"]),
        "dob": crypto.decrypt_field(data["dob_enc"]),
        "phone": crypto.decrypt_field(data["phone_enc"]),
        "member_id": crypto.decrypt_field(data["member_id_enc"])
        if data["member_id_enc"]
        else "",
        "contact_method": data["contact_method"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "appointment-scheduler"}


@app.get("/providers", response_model=ProvidersResponse)
def search_providers(
    specialty: str | None = None,
    taxonomy_code: str | None = None,
    city: str | None = None,
    state: str | None = None,
    first_name: str | None = None,
    last_name: str | None = None,
    limit: int = 10,
    conn: sqlite3.Connection = Depends(get_conn),
) -> ProvidersResponse:
    code = taxonomy_code or (taxonomy_code_for(specialty) if specialty else None)
    if specialty and not code:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown specialty {specialty!r}. Try cardiology, family medicine, internal medicine, or obstetrics & gynecology.",
        )
    client = CMSClient(db_path=db_path())
    result = client.search(
        taxonomy_code=code,
        city=city or DEFAULT_CITY,
        state=state or DEFAULT_STATE,
        first_name=first_name,
        last_name=last_name,
        limit=limit,
    )
    return ProvidersResponse(
        providers=[ProviderOut(**p) for p in result["providers"]],
        source=result["source"],
        note=result.get("note", ""),
    )


@app.get("/providers/{npi}/availability", response_model=AvailabilityResponse)
def provider_availability(
    npi: str,
    date_from: str | None = None,
    date_to: str | None = None,
    time_of_day: str | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> AvailabilityResponse:
    try:
        start = date.fromisoformat(date_from) if date_from else date.today()
        end = date.fromisoformat(date_to) if date_to else start + timedelta(days=13)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Dates must be YYYY-MM-DD.") from exc
    if (end - start).days > 30:
        raise HTTPException(status_code=400, detail="Date range is limited to 30 days.")
    return AvailabilityResponse(
        npi=npi,
        slots=generate_slots(conn, npi, start, end, time_of_day),
    )


@app.post("/holds", response_model=HoldOut)
def create_hold(body: HoldRequest, conn: sqlite3.Connection = Depends(get_conn)) -> HoldOut:
    now = time.time()
    taken = conn.execute(
        "SELECT appointment_id FROM appointments WHERE npi = ? AND slot = ? AND status = 'booked'",
        (body.npi, body.slot),
    ).fetchone()
    if taken:
        raise HTTPException(status_code=409, detail="That slot is already booked.")
    hold_id = f"HLD-{uuid.uuid4().hex[:8].upper()}"
    expires_at = now + HOLD_TTL_SECONDS
    conn.execute(
        "INSERT INTO holds (hold_id, npi, slot, session_id, expires_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (hold_id, body.npi, body.slot, body.session_id, expires_at, now),
    )
    conn.commit()
    logger.info("Hold %s created for session %s", hold_id, body.session_id)
    return HoldOut(
        hold_id=hold_id,
        npi=body.npi,
        slot=body.slot,
        expires_at=expires_at,
        expires_in_seconds=HOLD_TTL_SECONDS,
    )


@app.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(body: PatientCreate, conn: sqlite3.Connection = Depends(get_conn)) -> PatientOut:
    phone, err = slots.validate_phone(body.phone)
    if err:
        raise HTTPException(status_code=422, detail=err)
    dob_parsed, err = slots.validate_dob(body.dob)
    if err:
        raise HTTPException(status_code=422, detail=err)
    if body.contact_method not in ("text", "call", "email"):
        raise HTTPException(
            status_code=422, detail="contact_method must be text, call, or email."
        )
    patient_ref = f"PAT-{uuid.uuid4().hex[:8].upper()}"
    conn.execute(
        "INSERT INTO patients (patient_ref, full_name_enc, dob_enc, phone_enc, "
        "member_id_enc, contact_method, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            patient_ref,
            crypto.encrypt_field(body.full_name.strip()),
            crypto.encrypt_field(dob_parsed),
            crypto.encrypt_field(phone),
            crypto.encrypt_field(body.member_id.strip()) if body.member_id else None,
            body.contact_method,
            time.time(),
        ),
    )
    conn.commit()
    logger.info("Patient record created: %s", patient_ref)
    return PatientOut(
        patient_ref=patient_ref,
        name_masked=mask_name(body.full_name),
        dob_masked=mask_dob(dob_parsed),
        phone_masked=mask_phone(phone),
        member_id_masked=mask_member_id(body.member_id) if body.member_id else None,
        contact_method=body.contact_method,
    )


@app.get("/patients/{patient_ref}", response_model=PatientOut)
def get_patient(patient_ref: str, conn: sqlite3.Connection = Depends(get_conn)) -> PatientOut:
    record = _patient_row(conn, patient_ref)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    return PatientOut(
        patient_ref=record["patient_ref"],
        name_masked=mask_name(record["full_name"]),
        dob_masked=mask_dob(record["dob"]),
        phone_masked=mask_phone(record["phone"]),
        member_id_masked=mask_member_id(record["member_id"]) if record["member_id"] else None,
        contact_method=record["contact_method"],
    )


@app.patch("/patients/{patient_ref}", response_model=PatientOut)
def update_patient(
    patient_ref: str, body: PatientUpdate, conn: sqlite3.Connection = Depends(get_conn)
) -> PatientOut:
    record = _patient_row(conn, patient_ref)
    if record is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    if body.phone is not None:
        phone, err = slots.validate_phone(body.phone)
        if err:
            raise HTTPException(status_code=422, detail=err)
        conn.execute(
            "UPDATE patients SET phone_enc = ? WHERE patient_ref = ?",
            (crypto.encrypt_field(phone), patient_ref),
        )
    if body.contact_method is not None:
        if body.contact_method not in ("text", "call", "email"):
            raise HTTPException(
                status_code=422, detail="contact_method must be text, call, or email."
            )
        conn.execute(
            "UPDATE patients SET contact_method = ? WHERE patient_ref = ?",
            (body.contact_method, patient_ref),
        )
    conn.commit()
    return get_patient(patient_ref, conn)


@app.delete("/patients/{patient_ref}")
def delete_patient(patient_ref: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    cur = conn.execute("DELETE FROM patients WHERE patient_ref = ?", (patient_ref,))
    conn.commit()
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Patient not found.")
    logger.info("Patient record deleted: %s", patient_ref)
    return {"deleted": patient_ref}


@app.post("/appointments", response_model=AppointmentOut, status_code=201)
def book_appointment(body: AppointmentCreate, conn: sqlite3.Connection = Depends(get_conn)) -> AppointmentOut:
    # Idempotency first: the same key twice returns the original booking.
    existing = conn.execute(
        "SELECT appointment_id FROM idempotency_keys WHERE idempotency_key = ?",
        (body.idempotency_key,),
    ).fetchone()
    if existing:
        row = conn.execute(
            "SELECT * FROM appointments WHERE appointment_id = ?",
            (existing["appointment_id"],),
        ).fetchone()
        return _to_appointment_out(dict(row), duplicate=True)
    decision = policy.check_book(
        conn,
        confirmed=body.confirmed,
        idempotency_key=body.idempotency_key,
        npi=body.npi,
        slot=body.slot,
        hold_id=body.hold_id,
    )
    if not decision:
        raise HTTPException(status_code=409, detail=decision.reason)
    if _patient_row(conn, body.patient_ref) is None:
        raise HTTPException(status_code=404, detail="Patient reference not found.")
    count = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
    appointment_id = f"APT-{1001 + count}"
    now = time.time()
    conn.execute(
        "INSERT INTO appointments (appointment_id, patient_ref, npi, provider_name, "
        "slot, reason, contact_method, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'booked', ?, ?)",
        (
            appointment_id,
            body.patient_ref,
            body.npi,
            body.provider_name,
            body.slot,
            body.reason,
            body.contact_method,
            now,
            now,
        ),
    )
    conn.execute(
        "INSERT INTO idempotency_keys (idempotency_key, appointment_id, created_at) "
        "VALUES (?, ?, ?)",
        (body.idempotency_key, appointment_id, now),
    )
    if body.hold_id:
        conn.execute("DELETE FROM holds WHERE hold_id = ?", (body.hold_id,))
    conn.commit()
    logger.info("Appointment %s booked for %s", appointment_id, body.patient_ref)
    row = conn.execute(
        "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
    ).fetchone()
    return _to_appointment_out(dict(row))


@app.get("/appointments", response_model=list[AppointmentOut])
def list_appointments(
    patient_ref: str | None = None,
    conn: sqlite3.Connection = Depends(get_conn),
) -> list[AppointmentOut]:
    if patient_ref:
        rows = conn.execute(
            "SELECT * FROM appointments WHERE patient_ref = ? ORDER BY slot",
            (patient_ref,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM appointments ORDER BY slot").fetchall()
    return [_to_appointment_out(dict(r)) for r in rows]


@app.post("/appointments/{appointment_id}/cancel", response_model=AppointmentOut)
def cancel_appointment(appointment_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> AppointmentOut:
    decision = policy.check_cancel(conn, appointment_id=appointment_id)
    if not decision:
        raise HTTPException(status_code=409, detail=decision.reason)
    conn.execute(
        "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE appointment_id = ?",
        (time.time(), appointment_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
    ).fetchone()
    logger.info("Appointment %s cancelled", appointment_id)
    return _to_appointment_out(dict(row))


@app.post("/appointments/{appointment_id}/reschedule", response_model=AppointmentOut)
def reschedule_appointment(
    appointment_id: str, body: RescheduleRequest, conn: sqlite3.Connection = Depends(get_conn)
) -> AppointmentOut:
    row = conn.execute(
        "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Appointment not found.")
    decision = policy.check_reschedule(
        conn, appointment_id=appointment_id, npi=row["npi"], new_slot=body.new_slot
    )
    if not decision:
        raise HTTPException(status_code=409, detail=decision.reason)
    conn.execute(
        "UPDATE appointments SET slot = ?, updated_at = ? WHERE appointment_id = ?",
        (body.new_slot, time.time(), appointment_id),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM appointments WHERE appointment_id = ?", (appointment_id,)
    ).fetchone()
    logger.info("Appointment %s rescheduled to %s", appointment_id, body.new_slot)
    return _to_appointment_out(dict(row))


@app.get("/sessions")
def list_sessions(conn: sqlite3.Connection = Depends(get_conn)) -> list[dict]:
    rows = conn.execute(
        "SELECT session_id, label, summary, booking_state, created_at, updated_at "
        "FROM sessions ORDER BY updated_at DESC LIMIT 50"
    ).fetchall()
    out = []
    for row in rows:
        data = dict(row)
        data["booking_state"] = json.loads(data["booking_state"] or "{}")
        out.append(data)
    return out


@app.get("/sessions/{session_id}")
def get_session(session_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> dict:
    row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    data = dict(row)
    data["booking_state"] = json.loads(data["booking_state"] or "{}")
    messages = conn.execute(
        "SELECT role, content, created_at FROM session_messages "
        "WHERE session_id = ? ORDER BY id DESC LIMIT 50",
        (session_id,),
    ).fetchall()
    data["messages"] = [dict(m) for m in reversed(messages)]
    prefs = conn.execute(
        "SELECT pref_key, pref_value FROM preferences WHERE session_id = ?", (session_id,)
    ).fetchall()
    data["preferences"] = {p["pref_key"]: p["pref_value"] for p in prefs}
    return data


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """Talk to the agent. One endpoint for the UI: send a message, get the
    reply plus the updated booking state for the session card."""
    from .chat import get_chat_service

    service = get_chat_service()
    reply = service.chat(body.session_id or "", body.message)
    return ChatResponse(
        session_id=reply.session_id,
        reply=reply.reply,
        booking_stage=reply.booking_stage,
        booking_summary=reply.booking_summary,
        missing_slots=reply.missing_slots,
        preferences=reply.preferences,
        skills_activated=reply.skills_activated,
        cache=reply.cache,
        tools_used=reply.tools_used,
    )


# ---------------------------------------------------------------------------
# Static UI (single page, no build step)
# ---------------------------------------------------------------------------

_STATIC = Path(__file__).parent / "static"


@app.get("/")
def ui_index() -> FileResponse:
    return FileResponse(_STATIC / "index.html", media_type="text/html")


@app.get("/app.js")
def ui_js() -> FileResponse:
    return FileResponse(_STATIC / "app.js", media_type="application/javascript")


def main() -> None:
    import uvicorn

    port = int(os.environ.get("SCHEDULER_API_PORT", "8000"))
    # The MCP server subprocess (spawned by /chat) calls this API over
    # HTTP, so it must know the port. The demo sets SCHEDULER_API_URL
    # explicitly; derive it here when it isn't set.
    os.environ.setdefault(
        "SCHEDULER_API_URL", f"http://127.0.0.1:{port}"
    )
    uvicorn.run(
        "agent_appointment_scheduler.api:app",
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
