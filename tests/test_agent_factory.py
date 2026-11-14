"""Agent factory tests: tools load, session resumes, key fails fast."""

import os

import pytest

from agent_appointment_scheduler.agent import (
    EXPECTED_TOOLS,
    build_scripted_agent,
    list_loaded_tools,
)
from agent_appointment_scheduler.models import require_provider_key


def test_scripted_agent_loads_expected_tools(db_path):
    with build_scripted_agent("SES-A1", db_path=db_path) as (agent, hooks, skills_plugin):
        names = list(agent.tool_names)
        for expected in EXPECTED_TOOLS:
            assert expected in names, f"missing tool {expected}; loaded: {names}"


def test_agent_system_prompt_carries_session_id(db_path):
    with build_scripted_agent("SES-A2", db_path=db_path) as (agent, hooks, skills_plugin):
        assert "SES-A2" in (agent.system_prompt or "")


def test_agent_has_session_manager(db_path):
    with build_scripted_agent("SES-A3", db_path=db_path) as (agent, hooks, skills_plugin):
        manager = getattr(agent, "session_manager", None) or getattr(agent, "_session_manager", None)
        assert manager is not None
        assert manager.session_id == "SES-A3"


def test_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        require_provider_key("openai")


def test_unknown_provider_rejected():
    with pytest.raises(ValueError, match="Unknown provider"):
        require_provider_key("not-a-provider")
