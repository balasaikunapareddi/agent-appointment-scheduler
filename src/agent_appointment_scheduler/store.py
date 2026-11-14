"""SQLite persistence for the scheduling service.

One database file holds everything, so the FastAPI service, the MCP server
and the agent all read the same source of truth. Tables:

- ``providers_cache`` — CMS search results, TTL 24h.
- ``patients`` — patient records; sensitive fields are Fernet-encrypted at
  rest (see ``crypto``). The app only ever stores ciphertext here.
- ``appointments`` — booked appointments, keyed by patient reference.
- ``holds`` — slot holds with an expiry timestamp.
- ``idempotency_keys`` — key -> appointment id, for exactly-once booking.
- ``sessions`` / ``session_messages`` / ``preferences`` — session + memory
  (see ``sessions``); messages are stored REDACTED, never with raw PII.
- ``traces`` — structured per-turn traces for observability and replay.

Retention: sessions and their messages expire after 30 days
(``purge_expired``); patient records are deleted on request via the API.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

SESSION_TTL_SECONDS = 30 * 24 * 3600  # sessions expire after 30 days
PROVIDER_CACHE_TTL_SECONDS = 24 * 3600  # CMS results cached for 24 hours

SCHEMA = """
CREATE TABLE IF NOT EXISTS providers_cache (
    cache_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    fetched_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS patients (
    patient_ref TEXT PRIMARY KEY,
    full_name_enc TEXT NOT NULL,
    dob_enc TEXT NOT NULL,
    phone_enc TEXT NOT NULL,
    member_id_enc TEXT,
    contact_method TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS appointments (
    appointment_id TEXT PRIMARY KEY,
    patient_ref TEXT NOT NULL,
    npi TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    slot TEXT NOT NULL,
    reason TEXT,
    contact_method TEXT,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS holds (
    hold_id TEXT PRIMARY KEY,
    npi TEXT NOT NULL,
    slot TEXT NOT NULL,
    session_id TEXT NOT NULL,
    expires_at REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    appointment_id TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    label TEXT,
    summary TEXT,
    booking_state TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS session_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS preferences (
    session_id TEXT NOT NULL,
    pref_key TEXT NOT NULL,
    pref_value TEXT NOT NULL,
    PRIMARY KEY (session_id, pref_key)
);
CREATE TABLE IF NOT EXISTS traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    payload TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS strands_sessions (
    session_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS strands_messages (
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (session_id, agent_id, message_id)
);
CREATE TABLE IF NOT EXISTS strands_agents (
    session_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (session_id, agent_id)
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the database, creating the schema on first use (WAL mode)."""
    path = Path(db_path)
    if str(path) != ":memory:" and path.parent != Path("."):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def purge_expired(conn: sqlite3.Connection, now: float | None = None) -> dict:
    """Delete sessions (and their messages/preferences) older than the TTL.

    Patient records are NOT purged here — they are deleted only on request,
    per the data-minimization rule in the README.
    """
    now = now if now is not None else time.time()
    cutoff = now - SESSION_TTL_SECONDS
    cur = conn.execute(
        "SELECT session_id FROM sessions WHERE updated_at < ?", (cutoff,)
    )
    stale = [row[0] for row in cur.fetchall()]
    for session_id in stale:
        conn.execute(
            "DELETE FROM session_messages WHERE session_id = ?", (session_id,)
        )
        conn.execute("DELETE FROM preferences WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    return {"purged_sessions": len(stale)}


def dumps(payload: dict | list) -> str:
    return json.dumps(payload, ensure_ascii=False)


def loads(raw: str) -> dict | list:
    return json.loads(raw)
