"""Hook tests: guardrail cancellation, trace recording, cache metrics."""

import asyncio

import pytest
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent

from agent_appointment_scheduler.hooks import SchedulerHooks, TraceRecorder
from agent_appointment_scheduler.prompting import extract_cache_metrics


def _before_event(tool_name: str, args: dict) -> BeforeToolCallEvent:
    return BeforeToolCallEvent(
        invocation_context=None,
        agent=None,
        selected_tool=None,
        tool_use={"toolUseId": "t-1", "name": tool_name, "input": args},
        invocation_state={},
    )


def test_guardrail_cancels_unconfirmed_booking(db_path):
    hooks = SchedulerHooks(db_path=db_path, session_id="SES-H1")
    event = _before_event("book_appointment", {"confirmed": False, "session_id": "SES-H1"})
    asyncio.run(hooks.on_before_tool_call(event))
    assert event.cancel_tool, "unconfirmed booking must be cancelled by the hook"
    assert "confirm" in str(event.cancel_tool).lower()


def test_guardrail_allows_confirmed_booking(db_path):
    hooks = SchedulerHooks(db_path=db_path, session_id="SES-H1")
    event = _before_event("book_appointment", {"confirmed": True, "session_id": "SES-H1"})
    asyncio.run(hooks.on_before_tool_call(event))
    assert not event.cancel_tool


def test_guardrail_does_not_block_reads(db_path):
    hooks = SchedulerHooks(db_path=db_path, session_id="SES-H1")
    event = _before_event("search_providers", {"specialty": "cardiology"})
    asyncio.run(hooks.on_before_tool_call(event))
    assert not event.cancel_tool


def test_trace_recorder_round_trip(db_path):
    recorder = TraceRecorder(db_path)
    recorder.record("trace-1", "SES-H1", 1, {"tool": "search_providers", "outcome": "ok"})
    loaded = recorder.load("trace-1")
    assert len(loaded) == 1
    assert loaded[0]["tool"] == "search_providers"


def test_cache_metrics_openai_shape():
    usage = {
        "prompt_tokens": 2000,
        "completion_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 1500},
    }
    metrics = extract_cache_metrics("openai", "openai/gpt-4o-mini", usage)
    assert metrics.prompt_tokens == 2000
    assert metrics.cached_tokens == 1500
    assert metrics.cache_hit_rate == 0.75
    assert metrics.est_cost_saved_usd > 0


def test_cache_metrics_gemini_shape():
    usage = {
        "prompt_tokens": 3000,
        "completion_tokens": 50,
        "usage_metadata": {"cachedContentTokenCount": 2000},
    }
    metrics = extract_cache_metrics("gemini", "gemini/gemini-3.5-flash-lite", usage)
    assert metrics.cached_tokens == 2000
    assert metrics.cache_hit_rate == pytest.approx(2 / 3)


def test_cache_metrics_absent_is_zero():
    metrics = extract_cache_metrics("openai", "openai/gpt-4o-mini", {"prompt_tokens": 100})
    assert metrics.cached_tokens == 0
    assert metrics.cache_hit_rate == 0.0
    assert metrics.est_cost_saved_usd == 0.0
