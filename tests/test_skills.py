"""Skill tests: registry, selection, and per-turn activation."""

from agent_appointment_scheduler.chat import ChatService
from agent_appointment_scheduler.skills import (
    load_skills,
    registry_text,
    select_skills,
    skill_instructions,
)

EXPECTED = {"find-providers", "book-appointment", "reschedule-cancel", "session-memory"}


def test_all_four_skills_load():
    skills = load_skills()
    assert {s.name for s in skills} == EXPECTED
    for skill in skills:
        assert skill.description, f"{skill.name} has no description"
        assert skill.instructions.strip(), f"{skill.name} has no instructions"


def test_registry_lists_names_and_descriptions():
    text = registry_text()
    for name in EXPECTED:
        assert name in text
    # The registry is small — names + one line each, not full instructions.
    assert len(text) < 1000
    assert "taxonomy" not in text.lower()


def test_select_skills_search_turn():
    assert select_skills("I need a cardiologist near Minneapolis") == ["find-providers"]


def test_select_skills_booking_turn():
    picked = select_skills("Yes, book it", stage="confirming")
    assert "book-appointment" in picked


def test_select_skills_reschedule_turn():
    picked = select_skills("Can we move it to Thursday?", stage="booked")
    assert "reschedule-cancel" in picked


def test_skill_instructions_content():
    text = skill_instructions("find-providers")
    # The skill teaches the taxonomy DESCRIPTION CMS searches on, not the
    # code — CMS v2.1 has no taxonomy_code search field.
    assert "Cardiovascular Disease" in text
    assert "207RC0000X" not in text
    assert "book-appointment" not in skill_instructions("find-providers")
    booking = skill_instructions("book-appointment")
    assert "confirmed=true" in booking


def test_search_turn_loads_only_find_providers(db_path):
    service = ChatService(db_path=db_path, mock=True)
    session_id = "SES-SK1"
    service.chat(session_id, "I need a cardiologist near Minneapolis next week")
    agent = service._agents[session_id]
    _, _, skills_plugin = service._contexts[session_id]
    activated = skills_plugin.get_activated_skills(agent)
    assert activated == ["find-providers"], f"expected only find-providers, got {activated}"
    service.close()


def test_booking_turn_loads_booking_skill(db_path):
    service = ChatService(db_path=db_path, mock=True)
    session_id = "SES-SK2"
    service.chat(session_id, "I need a cardiologist near Minneapolis next week")
    service.chat(session_id, "the second one")
    service.chat(session_id, "Wednesday morning works")
    service.chat(session_id, "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147")
    agent = service._agents[session_id]
    _, _, skills_plugin = service._contexts[session_id]
    activated = skills_plugin.get_activated_skills(agent)
    assert "book-appointment" in activated, f"booking skill not activated: {activated}"
    assert "reschedule-cancel" not in activated
    service.close()
