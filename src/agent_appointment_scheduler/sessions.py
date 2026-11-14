"""Session + memory.

Three layers, all on one SQLite file (see ``store``):

- Working memory: the agent's in-context messages (Strands). Bounded by
  ``SlidingWindowConversationManager`` (40 turns) so context never grows
  without limit.
- Episodic memory: every turn persisted (redacted) in ``session_messages``
  plus a rolling summary. When history passes a threshold, older turns are
  summarized into ``sessions.summary`` and dropped from the context budget
  (they stay in the database for audit/replay).
- Semantic memory: durable user facts in ``preferences`` — time-of-day
  preference, city, contact method, usual provider. Extracted
  deterministically from what the user says, never by the model reading
  raw PII.

Strands-native persistence: ``SQLiteSessionRepository`` implements
``strands.session.SessionRepository``, so ``RepositorySessionManager``
stores and restores the agent's messages in the same SQLite file. The
smoke test in development confirmed a fresh agent with the same session
id resumes with the stored messages. The booking form state
(``slots.BookingForm``) lives in ``sessions.booking_state``.

PHI rule: stored messages are redacted before they are written, and the
summary/preferences hold only references and masked values. The model
never sees raw PII from memory.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone

from strands.session.session_repository import SessionRepository
from strands.types.session import Session, SessionAgent, SessionMessage

from . import store
from .phi import PIIRedactor
from .slots import BookingForm

SUMMARY_TURN_THRESHOLD = 12
SUMMARY_TOKEN_THRESHOLD = 3000
CONTEXT_BUDGET_TOKENS = 6000
WORKING_WINDOW_TURNS = 40


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4). Good enough for budgeting."""
    return max(1, len(text or "") // 4)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode(payload: dict) -> str:
    try:
        from strands.types.session import encode_bytes_values

        payload = encode_bytes_values(payload)
    except Exception:
        pass
    return json.dumps(payload, ensure_ascii=False)


def _decode(raw: str) -> dict:
    payload = json.loads(raw)
    try:
        from strands.types.session import decode_bytes_values

        payload = decode_bytes_values(payload)
    except Exception:
        pass
    return payload


def _from_dict(cls, data: dict):
    if is_dataclass(cls):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    return cls(**data)


class SQLiteSessionRepository(SessionRepository):
    """Strands-native session persistence on SQLite.

    Implements the ``SessionRepository`` interface the same way the file
    and S3 repositories do, but in the scheduler's own database, so
    sessions, messages, preferences and bookings share one file.
    """

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        return store.connect(self._db_path)

    def create_session(self, session: Session, **kwargs) -> Session:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO strands_sessions (session_id, payload, updated_at) "
                "VALUES (?, ?, ?)",
                (session.session_id, _encode(asdict(session)), time.time()),
            )
            conn.commit()
        finally:
            conn.close()
        return session

    def read_session(self, session_id: str, **kwargs) -> Session | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT payload FROM strands_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _from_dict(Session, _decode(row["payload"]))

    def create_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO strands_messages "
                "(session_id, agent_id, message_id, payload, updated_at) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    agent_id,
                    session_message.message_id,
                    _encode(asdict(session_message)),
                    time.time(),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def list_messages(self, session_id: str, agent_id: str, limit: int | None = None, offset: int = 0, **kwargs) -> list[SessionMessage]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT payload FROM strands_messages WHERE session_id = ? AND agent_id = ? "
                "ORDER BY message_id LIMIT ? OFFSET ?",
                (session_id, agent_id, limit if limit is not None else -1, offset),
            ).fetchall()
        finally:
            conn.close()
        return [_from_dict(SessionMessage, _decode(r["payload"])) for r in rows]

    def read_message(self, session_id: str, agent_id: str, message_id: int, **kwargs) -> SessionMessage | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT payload FROM strands_messages WHERE session_id = ? AND agent_id = ? AND message_id = ?",
                (session_id, agent_id, message_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _from_dict(SessionMessage, _decode(row["payload"]))

    def update_message(self, session_id: str, agent_id: str, session_message: SessionMessage, **kwargs) -> None:
        self.create_message(session_id, agent_id, session_message, **kwargs)

    def create_agent(self, session_id: str, session_agent: SessionAgent, **kwargs) -> None:
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO strands_agents (session_id, agent_id, payload, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, session_agent.agent_id, _encode(asdict(session_agent)), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def read_agent(self, session_id: str, agent_id: str, **kwargs) -> SessionAgent | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT payload FROM strands_agents WHERE session_id = ? AND agent_id = ?",
                (session_id, agent_id),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return _from_dict(SessionAgent, _decode(row["payload"]))

    def update_agent(self, session_id: str, session_agent: SessionAgent, **kwargs) -> None:
        self.create_agent(session_id, session_agent, **kwargs)


# ---------------------------------------------------------------------------
# The session store: summary, preferences, booking form, context budget
# ---------------------------------------------------------------------------

_PREFERENCE_PATTERNS: list[tuple[str, str]] = [
    (r"\bprefer(?:s|red)?\s+(?:the\s+)?mornings?\b", "time_of_day:morning"),
    (r"\bprefer(?:s|red)?\s+(?:the\s+)?afternoons?\b", "time_of_day:afternoon"),
    (r"\bmornings?\s+(?:work|are)?\s*(?:best|better|for me)?", "time_of_day:morning"),
    (r"\btext\s+me\b", "contact_method:text"),
    (r"\bcall\s+me\b", "contact_method:call"),
    (r"\bminneapolis\b", "city:Minneapolis"),
    (r"\bst\.?\s*paul\b", "city:Saint Paul"),
    (r"\brochester\b", "city:Rochester"),
]


def extract_preferences(text: str) -> dict[str, str]:
    """Deterministically pull durable preferences out of a user message."""
    found: dict[str, str] = {}
    lowered = (text or "").lower()
    for pattern, value in _PREFERENCE_PATTERNS:
        if re.search(pattern, lowered):
            key, _, val = value.partition(":")
            found[key] = val
    return found


class SessionStore:
    """Episodic + semantic memory for one database file."""

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path
        self._redactor = PIIRedactor()

    def _conn(self) -> sqlite3.Connection:
        return store.connect(self._db_path)

    def register_sensitive_values(self, *values: str) -> None:
        """Register literal values on this instance only (tests, one-offs)."""
        self._redactor.register(*values)

    def register_session_sensitive_values(self, session_id: str, *values: str) -> None:
        """Register literal sensitive values for a session — and make it stick.

        The literals are persisted in the session's booking state (as
        ``_sensitive_literals``) so every store instance redacts them,
        and every message already stored for the session is re-redacted
        in place. This matters because the user message that *gave* the
        name is stored before the patient record exists; without the
        rewrite, the raw name would sit in history forever.
        """
        values = [v for v in values if v]
        if not values:
            return
        self._redactor.register(*values)
        state = self._raw_booking_state(session_id)
        literals = sorted(set(state.get("_sensitive_literals", [])) | set(values))
        state["_sensitive_literals"] = literals
        self._save_raw_booking_state(session_id, state)
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT id, content FROM session_messages WHERE session_id = ?",
                (session_id,),
            ).fetchall()
            for row in rows:
                redacted = self._redactor.redact(row["content"])
                if redacted != row["content"]:
                    conn.execute(
                        "UPDATE session_messages SET content = ? WHERE id = ?",
                        (redacted, row["id"]),
                    )
            conn.commit()
        finally:
            conn.close()

    def _raw_booking_state(self, session_id: str) -> dict:
        session = self.get_session(session_id)
        return dict((session or {}).get("booking_state") or {})

    def _save_raw_booking_state(self, session_id: str, state: dict) -> None:
        self.create_session(session_id)
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE sessions SET booking_state = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(state), time.time(), session_id),
            )
            conn.commit()
        finally:
            conn.close()

    def _load_session_literals(self, session_id: str) -> None:
        for literal in self._raw_booking_state(session_id).get("_sensitive_literals", []):
            self._redactor.register(literal)

    # -- sessions -----------------------------------------------------------

    def create_session(self, session_id: str | None = None, label: str = "") -> str:
        session_id = session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
        now = time.time()
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR IGNORE INTO sessions "
                "(session_id, label, summary, booking_state, created_at, updated_at) "
                "VALUES (?, ?, '', '{}', ?, ?)",
                (session_id, label, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        return session_id

    def get_session(self, session_id: str) -> dict | None:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        data = dict(row)
        data["booking_state"] = json.loads(data["booking_state"] or "{}")
        return data

    def list_sessions(self, limit: int = 50) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT session_id, label, summary, created_at, updated_at "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def touch(self, session_id: str, label: str | None = None) -> None:
        conn = self._conn()
        try:
            if label:
                conn.execute(
                    "UPDATE sessions SET updated_at = ?, label = ? WHERE session_id = ?",
                    (time.time(), label, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (time.time(), session_id),
                )
            conn.commit()
        finally:
            conn.close()

    # -- messages (episodic) ------------------------------------------------

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Store one turn. Content is REDACTED first — raw PII never lands
        in stored history."""
        self.create_session(session_id)
        self._load_session_literals(session_id)
        redacted = self._redactor.redact(content)
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO session_messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, redacted, time.time()),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (time.time(), session_id),
            )
            conn.commit()
        finally:
            conn.close()

    def history(self, session_id: str, limit: int | None = None) -> list[dict]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT role, content, created_at FROM session_messages "
                "WHERE session_id = ? ORDER BY id LIMIT ?",
                (session_id, limit if limit is not None else -1),
            ).fetchall()
        finally:
            conn.close()
        return [dict(r) for r in rows]

    def message_count(self, session_id: str) -> int:
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM session_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        finally:
            conn.close()
        return row[0]

    # -- rolling summary ------------------------------------------------------

    def get_summary(self, session_id: str) -> str:
        session = self.get_session(session_id)
        return (session or {}).get("summary", "") or ""

    def maybe_summarize(self, session_id: str) -> bool:
        """Compact older turns into the stored summary when history passes
        the threshold. Deterministic and extractive: it keeps the first
        request, booking facts, and preferences — it never sends raw
        history to a model (that would leak PII into model context)."""
        messages = self.history(session_id)
        total_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        if len(messages) < SUMMARY_TURN_THRESHOLD and total_tokens < SUMMARY_TOKEN_THRESHOLD:
            return False
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        facts: list[str] = []
        for message in messages:
            content = message["content"]
            for pattern in (
                r"APT-\d+",
                r"PAT-[A-Z0-9]+",
                r"booked[^.]*",
                r"rescheduled[^.]*",
                r"provider[^.]*",
            ):
                found = re.findall(pattern, content, re.IGNORECASE)
                facts.extend(f[:120] for f in found[:2])
        summary = "Earlier in this conversation: " + (first_user[:200] if first_user else "booking help.")
        if facts:
            summary += " Key facts: " + "; ".join(sorted(set(facts))[:6]) + "."
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE sessions SET summary = ?, updated_at = ? WHERE session_id = ?",
                (summary, time.time(), session_id),
            )
            conn.commit()
        finally:
            conn.close()
        return True

    # -- preferences (semantic) ----------------------------------------------

    def set_preference(self, session_id: str, key: str, value: str) -> None:
        self.create_session(session_id)
        conn = self._conn()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO preferences (session_id, pref_key, pref_value) "
                "VALUES (?, ?, ?)",
                (session_id, key, value),
            )
            conn.commit()
        finally:
            conn.close()

    def get_preferences(self, session_id: str) -> dict[str, str]:
        conn = self._conn()
        try:
            rows = conn.execute(
                "SELECT pref_key, pref_value FROM preferences WHERE session_id = ?",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        return {r["pref_key"]: r["pref_value"] for r in rows}

    def record_preferences_from_text(self, session_id: str, text: str) -> dict[str, str]:
        found = extract_preferences(text)
        for key, value in found.items():
            self.set_preference(session_id, key, value)
        return found

    # -- booking form state ----------------------------------------------------

    def get_booking_form(self, session_id: str) -> BookingForm:
        session = self.get_session(session_id)
        state = (session or {}).get("booking_state") or {}
        form = BookingForm()
        for key, value in state.items():
            if hasattr(form, key):
                setattr(form, key, value)
        return form

    def save_booking_form(self, session_id: str, form: BookingForm) -> None:
        self.create_session(session_id)
        state = self._raw_booking_state(session_id)
        state.update(form.__dict__)
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE sessions SET booking_state = ?, updated_at = ? WHERE session_id = ?",
                (json.dumps(state), time.time(), session_id),
            )
            conn.commit()
        finally:
            conn.close()

    # -- context budget ---------------------------------------------------------

    def build_context(self, session_id: str, budget_tokens: int = CONTEXT_BUDGET_TOKENS) -> dict:
        """Assemble what goes into model context, within a token budget.

        Layout: summary first, then preferences, then the most recent turns
        that fit. Older turns are covered by the summary, not repeated.
        """
        summary = self.get_summary(session_id)
        preferences = self.get_preferences(session_id)
        messages = self.history(session_id)
        used = estimate_tokens(summary) + estimate_tokens(json.dumps(preferences))
        recent: list[dict] = []
        for message in reversed(messages):
            cost = estimate_tokens(message["content"])
            if used + cost > budget_tokens and recent:
                break
            used += cost
            recent.append(message)
        recent.reverse()
        return {
            "summary": summary,
            "preferences": preferences,
            "messages": recent,
            "estimated_tokens": used,
            "budget_tokens": budget_tokens,
            "total_messages": len(messages),
        }
