"""Fallback tests: model fallback, CMS fallback, bounded retries."""

import asyncio
import os
from collections.abc import AsyncIterable

import pytest

from agent_appointment_scheduler.models import (
    ModelWithFallback,
    build_model_with_fallback,
    fallback_provider_for,
)
from agent_appointment_scheduler.scripted_model import ScriptedModel


class FailingModel(ScriptedModel):
    """Raises before producing any content."""

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):  # type: ignore[override]
        raise RuntimeError("primary provider is down")
        yield  # pragma: no cover - makes this an async generator


async def _collect(stream: AsyncIterable) -> list:
    return [event async for event in stream]


def test_fallback_used_when_primary_fails_before_content():
    primary = FailingModel()
    fallback = ScriptedModel()
    model = ModelWithFallback(primary, fallback, primary_id="p", fallback_id="f")
    events = asyncio.run(_collect(model.stream([{"role": "user", "content": [{"text": "hi"}]}])))
    assert model.fell_back is True
    assert events, "fallback should produce events"


def test_fallback_not_used_when_primary_succeeds():
    primary = ScriptedModel()
    fallback = ScriptedModel()
    model = ModelWithFallback(primary, fallback, primary_id="p", fallback_id="f")
    asyncio.run(_collect(model.stream([{"role": "user", "content": [{"text": "hi"}]}])))
    assert model.fell_back is False


def test_no_fallback_without_other_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    assert fallback_provider_for("openai") is None


def test_fallback_provider_when_key_present(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert fallback_provider_for("openai") == "gemini"
    assert fallback_provider_for("gemini") == "openai"
