"""The Strands scheduling agent.

One orchestrator agent with deterministic tools — no subagents. The
agent talks to the user, loads skills, and calls MCP tools; every
decision (slot free, confirmation, hold, idempotency) lives in the
service and in ``policy``/``slots``/``planner``.

By default the agent runs against a real model (LiteLLM) and needs a
key — ``provider="openai"`` uses ``openai/gpt-4o-mini`` (OPENAI_API_KEY),
``provider="gemini"`` uses ``gemini/gemini-3.5-flash-lite`` on the free
tier (GEMINI_API_KEY). If the primary provider fails before a turn
produces content, ``ModelWithFallback`` retries the turn on the other
provider (only when its key is set). The scripted mock model exists
only for the tests and the explicit offline demo (``--mock``); the real
path never silently falls back to it. A missing key fails fast with a
clear message.

Session: ``RepositorySessionManager`` on the SQLite repository, so a
fresh agent with the same session id resumes with the stored messages.
The system prompt is composed static-first (see ``prompting``) and the
skill registry is injected by the native ``AgentSkills`` plugin.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from mcp import StdioServerParameters, stdio_client
from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import RepositorySessionManager
from strands.tools.mcp import MCPClient
from strands.vended_plugins.skills import AgentSkills

from . import store
from .hooks import SchedulerHooks
from .models import PROVIDERS, build_model_with_fallback, require_provider_key
from .prompting import compose_system_prompt
from .scripted_model import ScriptedModel
from .sessions import (
    WORKING_WINDOW_TURNS,
    SessionStore,
    SQLiteSessionRepository,
)
from .skills import SKILLS_DIR

logger = logging.getLogger(__name__)

EXPECTED_TOOLS = [
    "search_providers",
    "get_availability",
    "create_patient_record",
    "update_patient_contact",
    "choose_provider",
    "choose_slot",
    "set_visit_details",
    "get_booking_status",
    "get_plan",
    "book_appointment",
    "list_appointments",
    "cancel_appointment",
    "reschedule_appointment",
    "skills",
]


def create_mcp_client() -> MCPClient:
    """Build the MCP client that spawns the scheduler server."""

    def _transport():
        return stdio_client(
            StdioServerParameters(
                command=sys.executable,
                args=["-m", "agent_appointment_scheduler.mcp_server"],
                env={**os.environ},
            )
        )

    return MCPClient(_transport)


def list_loaded_tools(client: MCPClient) -> list[tuple[str, str]]:
    """Return (name, one-line description) for every tool on the server."""
    tools = client.list_tools_sync()
    out = []
    for tool in tools:
        description = (tool.tool_spec.get("description") or "").strip()
        one_line = description.splitlines()[0] if description else "(no description)"
        out.append((tool.tool_name, one_line))
    return out


def print_tool_list(tools: list[tuple[str, str]]) -> None:
    print(f"Loaded {len(tools)} tools from the MCP server:")
    for name, description in tools:
        print(f"  - {name}: {description}")


@contextmanager
def scheduler_agent(
    session_id: str,
    db_path: str = "scheduler.db",
    provider: str = "openai",
    model: Any | None = None,
    callback_handler: Any | None = None,
) -> Iterator[tuple[Agent, SchedulerHooks, AgentSkills]]:
    """Start the MCP server, load its tools, and yield a ready agent.

    ``model`` is for the scripted model in tests and the offline demo
    only; with no model given, the agent runs against a real model and
    requires that provider's key (see :func:`require_provider_key`).
    The MCP client is owned by the Strands Agent: constructing the agent
    starts the server subprocess, so the tool list is printed after the
    agent is built and before any request.
    """
    logging.basicConfig(level=logging.INFO)
    store.connect(db_path).close()
    session_store = SessionStore(db_path)
    session_store.create_session(session_id)
    context = session_store.build_context(session_id)
    system_prompt = compose_system_prompt(
        session_id,
        summary=context["summary"],
        preferences=context["preferences"],
    )
    client = create_mcp_client()
    if model is None:
        require_provider_key(provider)
        agent_model: Any = build_model_with_fallback(provider)
        model_id = PROVIDERS[provider]["model_id"]
    else:
        agent_model = model
        model_id = "scripted-mock"
    hooks = SchedulerHooks(
        db_path=db_path,
        session_id=session_id,
        provider=provider,
        model_id=model_id,
    )
    skills_plugin = AgentSkills(skills=[str(SKILLS_DIR)])
    repository = SQLiteSessionRepository(db_path)
    session_manager = RepositorySessionManager(session_id, repository)
    agent = Agent(
        model=agent_model,
        tools=[client],
        plugins=[skills_plugin],
        hooks=[hooks],
        system_prompt=system_prompt,
        session_manager=session_manager,
        conversation_manager=SlidingWindowConversationManager(
            window_size=WORKING_WINDOW_TURNS
        ),
        state={"session_id": session_id},
        callback_handler=callback_handler,
    )
    # The session manager restores stored messages on construction. If
    # this session only exists in the app's own store (e.g. it was
    # created through the API), load its history from there instead.
    if not agent.messages:
        history = context["messages"]
        if history:
            agent.messages = [
                {
                    "role": message["role"],
                    "content": [{"text": message["content"]}],
                }
                for message in history
            ]
    try:
        print_tool_list(list_loaded_tools(client))
        yield agent, hooks, skills_plugin
    finally:
        client.stop(None, None, None)


def build_scripted_agent(session_id: str, db_path: str = "scheduler.db"):
    """Convenience helper for tests and the offline demo: agent on the mock model."""
    return scheduler_agent(session_id=session_id, db_path=db_path, model=ScriptedModel())


__all__ = [
    "EXPECTED_TOOLS",
    "PROVIDERS",
    "build_scripted_agent",
    "scheduler_agent",
]
