"""SQLite persistence for the scheduling service.

One database holds everything, so the FastAPI service, the MCP server
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

Where the database lives
-----------------------
Local file (default): plain sqlite3, exactly as before — this is what
tests and local runs use.

Remote (hosted): set ``SCHEDULER_DATABASE_URL`` to a ``libsql://`` URL
and ``SCHEDULER_DATABASE_AUTH_TOKEN`` to its token (Turso free tier:
no card, 5 GB). Render's free tier wipes local files on every
spin-down, restart and redeploy, so a local ``scheduler.db`` is a cache
that empties itself — bookings vanish, appointment ids restart at
APT-1001, and the double-booking guard has nothing to check against.
The hosted database is what makes a booking survive a restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

SESSION_TTL_SECONDS = 30 * 24 * 3600  # sessions expire after 30 days
PROVIDER_CACHE_TTL_SECONDS = 24 * 3600  # CMS results cached for 24 hours

REMOTE_URL_ENV = "SCHEDULER_DATABASE_URL"
REMOTE_TOKEN_ENV = "SCHEDULER_DATABASE_AUTH_TOKEN"

# Remote schema is ensured once per process per URL: every request opens
# a fresh connection, and re-running the schema over the network each
# time would be pure overhead. (Local files keep the old behaviour of
# ensuring the schema on every connect — it is cheap and keeps temp-DB
# tests hermetic.)
_remote_schema_done: set[str] = set()

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


class _Row:
    """A ``sqlite3.Row`` look-alike for rows from the libsql adapter.

    Supports index and name access, ``.keys()``, ``.get()``, iteration
    over values, and ``dict(row)`` — everything the codebase asks of
    ``sqlite3.Row``. (libsql returns plain tuples, so the adapter has to
    add the names back.)
    """

    __slots__ = ("_values", "_names")

    def __init__(self, values: tuple, names: tuple[str, ...]) -> None:
        self._values = values
        self._names = names

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                return self._values[self._names.index(key)]
            except ValueError:
                raise IndexError(f"No item with that key: {key!r}") from None
        return self._values[key]

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def keys(self):
        return list(self._names)

    def get(self, key, default=None):
        try:
            return self[key]
        except (IndexError, KeyError):
            return default

    def __repr__(self) -> str:
        return f"_Row({dict(self)!r})"


class _Cursor:
    """Wraps a libsql cursor; ``fetchone``/``fetchall`` return ``_Row``."""

    def __init__(self, cursor) -> None:
        self._cursor = cursor

    @property
    def description(self):
        return self._cursor.description

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def _names(self) -> tuple[str, ...]:
        return tuple(col[0] for col in (self._cursor.description or ()))

    def _wrap(self, row):
        if row is None:
            return None
        return _Row(tuple(row), self._names())

    def fetchone(self):
        return self._wrap(self._cursor.fetchone())

    def fetchall(self):
        return [self._wrap(row) for row in self._cursor.fetchall()]

    def fetchmany(self, size=None):
        rows = (
            self._cursor.fetchmany(size)
            if size is not None
            else self._cursor.fetchmany()
        )
        return [self._wrap(row) for row in rows]

    def __iter__(self):
        for row in self._cursor:
            yield self._wrap(row)

    def close(self):
        self._cursor.close()


class _LibsqlConnection:
    """Wraps a libsql connection so it quacks like ``sqlite3.Connection``.

    The codebase only uses ``execute``/``executemany``/``executescript``,
    ``commit``/``rollback`` and ``close`` (plus the context-manager
    protocol), all of which libsql provides — the only gap is the row
    shape, which ``_Cursor`` closes.
    """

    def __init__(self, connection) -> None:
        self._conn = connection

    def execute(self, sql, parameters=()):
        return _Cursor(self._conn.execute(sql, parameters))

    def executemany(self, sql, seq_of_parameters):
        return _Cursor(self._conn.executemany(sql, seq_of_parameters))

    def executescript(self, sql):
        return _Cursor(self._conn.executescript(sql))

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    @property
    def in_transaction(self):
        return self._conn.in_transaction


def connect(db_path: str | Path) -> Any:
    """Open the database, creating the schema on first use (WAL mode).

    Local file (default): plain sqlite3, exactly as before.
    Remote (Turso): when ``SCHEDULER_DATABASE_URL`` is set to a
    ``libsql://`` URL, connect to the hosted database instead — this is
    what keeps bookings alive across Render free-tier spin-downs and
    restarts. ``:memory:`` always stays local.
    """
    url = os.environ.get(REMOTE_URL_ENV, "").strip()
    if url and str(db_path) != ":memory:":
        import libsql

        conn = _LibsqlConnection(
            libsql.connect(
                url,
                auth_token=os.environ.get(REMOTE_TOKEN_ENV, ""),
                timeout=30.0,
            )
        )
        if url not in _remote_schema_done:
            conn.executescript(SCHEMA)
            conn.commit()
            _remote_schema_done.add(url)
        return conn
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
