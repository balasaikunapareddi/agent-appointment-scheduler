"""Lifecycle hooks (Strands ``HookProvider``).

One provider, three jobs — all deterministic, all PII-safe:

1. Structured logging with PII redaction: every tool call is logged
   with its name, redacted arguments, duration and outcome. Raw
   sensitive values never reach a log line.
2. Token/cost accounting: per-turn usage is recorded into the traces
   table, including cache metrics (see ``prompting``). The demo prints
   a per-turn accounting summary, and the UI shows it in the status area.
3. Guardrail checks on tool calls: before a consequential tool runs
   (book / cancel / reschedule), the hook re-checks the policy in code
   and cancels the call when it fails — a second gate behind the
   service-side policy. The cancellation message goes back to the model
   as a tool error, so the agent explains it instead of retrying blindly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from strands.hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeToolCallEvent,
    MessageAddedEvent,
)
from strands.hooks.registry import HookProvider, HookRegistry

from . import store
from .phi import PIIRedactor
from .prompting import CacheMetrics, extract_cache_metrics

logger = logging.getLogger(__name__)

CONSEQUENTIAL_TOOLS = {"book_appointment", "cancel_appointment", "reschedule_appointment"}


class TraceRecorder:
    """Append structured per-turn traces to SQLite (for replay + audit)."""

    def __init__(self, db_path: str = "scheduler.db") -> None:
        self._db_path = db_path

    def record(
        self, trace_id: str, session_id: str, turn_index: int, payload: dict
    ) -> None:
        conn = store.connect(self._db_path)
        try:
            conn.execute(
                "INSERT INTO traces (trace_id, session_id, turn_index, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (trace_id, session_id, turn_index, json.dumps(payload), time.time()),
            )
            conn.commit()
        finally:
            conn.close()

    def load(self, trace_id: str) -> list[dict]:
        conn = store.connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT payload FROM traces WHERE trace_id = ? ORDER BY turn_index",
                (trace_id,),
            ).fetchall()
        finally:
            conn.close()
        return [json.loads(r["payload"]) for r in rows]


class SchedulerHooks(HookProvider):
    """Observability + guardrail hooks for the scheduling agent."""

    def __init__(
        self,
        db_path: str = "scheduler.db",
        session_id: str = "",
        provider: str = "openai",
        model_id: str = "",
    ) -> None:
        self._db_path = db_path
        self.session_id = session_id
        self.provider = provider
        self.model_id = model_id
        self._redactor = PIIRedactor()
        self._recorder = TraceRecorder(db_path)
        self._turn_index = 0
        self._turn_tools: list[dict] = []
        self._turn_started_at: float = 0.0
        self.last_metrics: CacheMetrics | None = None
        self.turn_traces: list[dict] = []

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self.on_before_tool_call)
        registry.add_callback(AfterToolCallEvent, self.on_after_tool_call)
        registry.add_callback(AfterModelCallEvent, self.on_after_model_call)
        registry.add_callback(MessageAddedEvent, self.on_message_added)

    # -- guardrail: cancel consequential calls that break policy ----------

    async def on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = tool_use.get("name", "")
        args = tool_use.get("input") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        logger.info(
            "tool_call session=%s tool=%s args=%s",
            self.session_id,
            name,
            self._redactor.redact(json.dumps(args)),
        )
        if name == "book_appointment" and not args.get("confirmed"):
            event.cancel_tool = (
                "Policy: booking requires explicit user confirmation. "
                "Show the masked summary and ask the user to confirm first."
            )
            logger.warning("tool_call cancelled by policy: %s without confirmation", name)

    # -- observability -------------------------------------------------------

    async def on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        tool_use = event.tool_use or {}
        name = tool_use.get("name", "")
        outcome = "error" if event.exception else "ok"
        entry = {
            "tool": name,
            "duration_ms": round((event.duration or 0) * 1000, 1)
            if isinstance(event.duration, (int, float))
            else 0,
            "outcome": outcome,
            "error": str(event.exception)[:200] if event.exception else "",
        }
        self._turn_tools.append(entry)
        logger.info(
            "tool_result session=%s tool=%s outcome=%s duration_ms=%s",
            self.session_id,
            name,
            outcome,
            entry["duration_ms"],
        )

    async def on_after_model_call(self, event: AfterModelCallEvent) -> None:
        usage = self._usage_from_event(event)
        if usage:
            self.last_metrics = extract_cache_metrics(self.provider, self.model_id, usage)
            logger.info(
                "model_call session=%s prompt_tokens=%d cached_tokens=%d hit_rate=%.2f",
                self.session_id,
                self.last_metrics.prompt_tokens,
                self.last_metrics.cached_tokens,
                self.last_metrics.cache_hit_rate,
            )

    async def on_message_added(self, event: MessageAddedEvent) -> None:
        message = event.message or {}
        role = message.get("role", "?")
        logger.debug("message_added session=%s role=%s", self.session_id, role)

    # -- per-turn trace -------------------------------------------------------

    def begin_turn(self) -> None:
        self._turn_index += 1
        self._turn_tools = []
        self._turn_started_at = time.time()

    def end_turn(self, user_message: str, reply: str, trace_id: str) -> dict:
        # The patient literals for this session live in the booking state
        # (written by the MCP tools during the turn). Load them before
        # redacting, or the trace would keep the raw name from the user
        # message that gave it.
        try:
            from .sessions import SessionStore

            store = SessionStore(self._db_path)
            literals = store._raw_booking_state(self.session_id).get(
                "_sensitive_literals", []
            )
            self._redactor.register(*literals)
        except Exception:
            pass
        payload = {
            "turn_index": self._turn_index,
            "user_message": self._redactor.redact(user_message),
            "reply": self._redactor.redact(reply),
            "tools": self._turn_tools,
            "latency_ms": round((time.time() - self._turn_started_at) * 1000, 1),
            "cache": self.last_metrics.as_dict() if self.last_metrics else None,
        }
        self.turn_traces.append(payload)
        self._recorder.record(trace_id, self.session_id, self._turn_index, payload)
        return payload

    @staticmethod
    def _usage_from_event(event: AfterModelCallEvent) -> dict | None:
        response = getattr(event, "stop_response", None)
        candidates = []
        if isinstance(response, dict):
            candidates.append(response.get("usage") or response.get("metadata") or {})
            meta = response.get("metadata") or {}
            candidates.append(meta.get("usage") or {})
        invocation_state = getattr(event, "invocation_state", None) or {}
        if isinstance(invocation_state, dict):
            candidates.append(invocation_state.get("usage") or {})
        for candidate in candidates:
            if isinstance(candidate, dict) and (
                candidate.get("prompt_tokens") or candidate.get("input_tokens")
            ):
                usage = dict(candidate)
                if "prompt_tokens" not in usage and "input_tokens" in usage:
                    usage["prompt_tokens"] = usage["input_tokens"]
                if "completion_tokens" not in usage and "output_tokens" in usage:
                    usage["completion_tokens"] = usage["output_tokens"]
                return usage
        return None
