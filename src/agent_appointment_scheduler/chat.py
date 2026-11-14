"""Chat service: the HTTP-facing front of the agent.

``POST /chat`` (and the demo) talk to the agent through this service.
One agent per session, cached while the process lives; a fresh process
resumes from the Strands session manager, so nothing is lost on a
restart. Every turn is recorded in the session store (redacted), the
booking state and preferences are updated, and the reply carries the
booking card state plus the turn's cache metrics for the UI status area.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .agent import scheduler_agent
from .prompting import extract_cache_metrics
from .scripted_model import ScriptedModel
from .sessions import SessionStore


def result_text(result) -> str:
    """Pull the final assistant text out of an AgentResult."""
    if hasattr(result, "message"):
        message = result.message
    elif isinstance(result, dict):
        message = result.get("message", {})
    else:
        return str(result)
    if isinstance(message, dict):
        parts = [
            block.get("text", "")
            for block in message.get("content", [])
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(part for part in parts if part).strip()
    return str(message)


@dataclass
class ChatReply:
    session_id: str
    reply: str
    booking_stage: str = ""
    booking_summary: str = ""
    missing_slots: list = field(default_factory=list)
    preferences: dict = field(default_factory=dict)
    skills_activated: list = field(default_factory=list)
    cache: dict = field(default_factory=dict)
    tools_used: list = field(default_factory=list)


class ChatService:
    """One agent per session, with per-turn accounting."""

    def __init__(
        self,
        db_path: str = "scheduler.db",
        provider: str = "openai",
        mock: bool = False,
    ) -> None:
        self._db_path = db_path
        self._provider = provider
        self._mock = mock
        self._store = SessionStore(db_path)
        self._agents: dict[str, object] = {}
        self._contexts: dict[str, object] = {}

    def _get_agent(self, session_id: str):
        if session_id not in self._agents:
            model = ScriptedModel() if self._mock else None
            ctx = scheduler_agent(
                session_id=session_id,
                db_path=self._db_path,
                provider=self._provider,
                model=model,
            )
            agent, hooks, skills_plugin = ctx.__enter__()
            self._agents[session_id] = agent
            self._contexts[session_id] = (ctx, hooks, skills_plugin)
        return self._agents[session_id], self._contexts[session_id]

    def close(self) -> None:
        for ctx, _, _ in self._contexts.values():
            ctx.__exit__(None, None, None)
        self._agents.clear()
        self._contexts.clear()

    def chat(self, session_id: str, message: str) -> ChatReply:
        session_id = session_id or f"SES-{uuid.uuid4().hex[:8].upper()}"
        agent, (ctx, hooks, skills_plugin) = self._get_agent(session_id)
        self._store.record_preferences_from_text(session_id, message)
        self._store.append_message(session_id, "user", message)
        hooks.begin_turn()
        result = agent(message)
        reply = result_text(result)
        self._store.append_message(session_id, "assistant", reply)
        self._store.maybe_summarize(session_id)
        trace = hooks.end_turn(message, reply, trace_id=f"trace-{session_id}")
        form = self._store.get_booking_form(session_id)
        summary = form.summary()
        usage = self._usage_from_result(result)
        metrics = None
        if hooks.last_metrics is not None and usage:
            metrics = extract_cache_metrics(
                self._provider,
                hooks.model_id,
                usage,
            )
        elif hooks.last_metrics is not None:
            metrics = hooks.last_metrics
        return ChatReply(
            session_id=session_id,
            reply=reply,
            booking_stage=form.stage,
            booking_summary=summary.readable(),
            missing_slots=summary.missing,
            preferences=self._store.get_preferences(session_id),
            skills_activated=skills_plugin.get_activated_skills(agent),
            cache=metrics.as_dict() if metrics else {},
            tools_used=[t["tool"] for t in trace.get("tools", [])],
        )

    def session_state(self, session_id: str) -> dict:
        form = self._store.get_booking_form(session_id)
        summary = form.summary()
        session = self._store.get_session(session_id) or {}
        return {
            "session_id": session_id,
            "label": session.get("label", ""),
            "stage": form.stage,
            "summary": summary.model_dump(),
            "readable_summary": summary.readable(),
            "missing": summary.missing,
            "next_question": form.next_question(),
            "preferences": self._store.get_preferences(session_id),
            "rolling_summary": session.get("summary", ""),
            "message_count": self._store.message_count(session_id),
        }

    def list_sessions(self) -> list[dict]:
        return self._store.list_sessions()

    @staticmethod
    def _usage_from_result(result) -> dict | None:
        metadata = getattr(result, "metadata", None) or {}
        if isinstance(metadata, dict):
            usage = metadata.get("usage")
            if isinstance(usage, dict):
                return usage
        metrics = getattr(result, "metrics", None) or {}
        if isinstance(metrics, dict):
            usage = metrics.get("usage")
            if isinstance(usage, dict):
                return usage
        return None


_CHAT_SERVICE: ChatService | None = None


def get_chat_service() -> ChatService:
    """Process-wide chat service (lazy singleton for the API).

    Provider/mock come from the environment: SCHEDULER_PROVIDER
    (openai|gemini, default openai), SCHEDULER_MOCK=1 for the scripted
    model. The MCP server subprocess inherits the same environment, so
    SCHEDULER_DB_PATH / SCHEDULER_API_TRANSPORT apply to it too.
    """
    global _CHAT_SERVICE
    if _CHAT_SERVICE is None:
        import os

        from .models import load_dotenv

        load_dotenv()
        # The MCP server subprocess calls the API over HTTP; make sure it
        # knows the port this process serves on.
        os.environ.setdefault(
            "SCHEDULER_API_URL",
            f"http://127.0.0.1:{os.environ.get('SCHEDULER_API_PORT', '8000')}",
        )
        _CHAT_SERVICE = ChatService(
            db_path=os.environ.get("SCHEDULER_DB_PATH", "scheduler.db"),
            provider=os.environ.get("SCHEDULER_PROVIDER", "openai"),
            mock=os.environ.get("SCHEDULER_MOCK", "") == "1",
        )
    return _CHAT_SERVICE
