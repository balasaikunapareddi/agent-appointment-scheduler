"""Prompt-caching tests: static prefix stable, dynamic content after it."""

import pytest

from agent_appointment_scheduler.prompting import (
    BASE_SYSTEM_PROMPT,
    compose_system_prompt,
    extract_cache_metrics,
    static_prefix,
)
from agent_appointment_scheduler.skills import registry_text


def test_static_prefix_is_stable_and_small():
    first = static_prefix()
    second = static_prefix()
    assert first == second
    assert first == BASE_SYSTEM_PROMPT.rstrip()
    # The base prompt stays small: the skills live in files, not in it.
    assert len(first) < 2000


def test_static_prefix_identical_across_two_turns():
    # Turn 1: fresh session. Turn 2: same session, summary + preferences now set.
    turn1 = compose_system_prompt("SES-C1")
    turn2 = compose_system_prompt(
        "SES-C1",
        summary="Earlier: booked APT-1001 with Dr Abbasi on 2026-11-18T09:00.",
        preferences={"time_of_day": "morning", "city": "Minneapolis"},
    )
    prefix = static_prefix()
    assert turn1.startswith(prefix)
    assert turn2.startswith(prefix)
    # The static prefix is byte-identical in both.
    assert turn1[: len(prefix)] == turn2[: len(prefix)]
    # Dynamic content comes after the static prefix, never inside it.
    assert "SES-C1" in turn1[len(prefix):]
    assert "APT-1001" in turn2[len(prefix):]
    assert "APT-1001" not in turn1


def test_registry_is_stable_across_turns():
    assert registry_text() == registry_text()


def test_compose_includes_session_id_after_prefix():
    prompt = compose_system_prompt("SES-C9")
    assert "Session ID: SES-C9" in prompt
    assert prompt.index(static_prefix()) == 0
    assert prompt.index("Session ID: SES-C9") > len(static_prefix())


def test_compose_includes_todays_date_after_prefix():
    # The live Gemini demo booked a slot in May 2025 because the model
    # had no current date and guessed "next week" from training
    # knowledge. The date now comes from the system clock, in the
    # dynamic part of the prompt — never inside the static prefix.
    from datetime import date

    prompt = compose_system_prompt("SES-C9")
    assert f"Today is {date.today():%A, %B %d, %Y}" in prompt
    assert prompt.index("Today is") > len(static_prefix())
    assert "Today is" not in static_prefix()


def test_cache_metrics_from_usage():
    usage = {"prompt_tokens": 4000, "completion_tokens": 120, "prompt_tokens_details": {"cached_tokens": 3000}}
    metrics = extract_cache_metrics("openai", "openai/gpt-4o-mini", usage)
    assert metrics.cache_hit_rate == 0.75
    # 3000 cached tokens at $0.15/M, saving 90% of their price.
    assert metrics.est_cost_saved_usd == pytest.approx(3000 * 0.15 * 0.9 / 1_000_000)
