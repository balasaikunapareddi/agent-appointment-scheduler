"""Crisis screen tests: emergencies get the fixed reply, nothing else.

The screen is deterministic code, not the model: a crisis turn must
never call a tool, never book, and never reach the model at all. The
control cases prove the screen is narrow — ordinary complaints and
routine requests still flow to the agent.
"""

import pytest

from agent_appointment_scheduler import store
from agent_appointment_scheduler.chat import ChatService
from agent_appointment_scheduler.crisis import (
    EMERGENCY_REPLY,
    SELF_HARM_REPLY,
    crisis_reply,
    detect_crisis,
)


@pytest.mark.parametrize(
    "message",
    [
        "I'm having chest pain right now",
        "My chest hurts really bad",
        "I can't breathe",
        "My baby is not breathing",
        "He passed out and won't wake up",
        "I think I'm having a stroke, my face is drooping",
        "I took too many pills",
        "She's having a seizure",
        "I'm bleeding heavily and it won't stop bleeding",
        "I'm choking",
    ],
)
def test_detect_crisis_emergency(message):
    assert detect_crisis(message) == "emergency"


@pytest.mark.parametrize(
    "message",
    [
        "I want to kill myself",
        "I don't want to live anymore",
        "I've been thinking about suicide",
    ],
)
def test_detect_crisis_self_harm(message):
    assert detect_crisis(message) == "self_harm"


@pytest.mark.parametrize(
    "message",
    [
        "I have a mild headache, need a doctor next week",
        "I have pain in my knee",
        "My back hurts and I need a checkup",
        "I need a cardiologist near Minneapolis next week",
        "I had a stroke last year and need a follow-up",  # see note below
    ],
)
def test_detect_crisis_control(message):
    # NOTE: the last case is a known over-trigger the list accepts: the
    # phrase "stroke" is not in the list, but "having a stroke" is — and
    # this sentence says "had a stroke", which does not match. If the
    # list ever gains a bare "stroke", this case must be revisited.
    assert detect_crisis(message) is None


def test_chest_pain_gets_fixed_reply_and_books_nothing(db_path):
    service = ChatService(db_path=db_path, mock=True)
    reply = service.chat("SES-C1", "I'm having chest pain right now")
    assert reply.reply == EMERGENCY_REPLY
    assert "911" in reply.reply
    assert reply.tools_used == []
    assert reply.booking_stage == ""
    conn = store.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
    finally:
        conn.close()
    service.close()


def test_baby_not_breathing_gets_fixed_reply(db_path):
    service = ChatService(db_path=db_path, mock=True)
    reply = service.chat("SES-C2", "My baby is not breathing, please help")
    assert reply.reply == EMERGENCY_REPLY
    assert reply.tools_used == []
    service.close()


def test_self_harm_gets_988_reply(db_path):
    service = ChatService(db_path=db_path, mock=True)
    reply = service.chat("SES-C3", "I want to kill myself")
    assert reply.reply == SELF_HARM_REPLY
    assert "988" in reply.reply
    assert reply.tools_used == []
    service.close()


def test_crisis_mid_booking_blocks_the_booking(db_path):
    service = ChatService(db_path=db_path, mock=True)
    service.chat("SES-C4", "I need a cardiologist near Minneapolis next week")
    service.chat("SES-C4", "the first one")
    reply = service.chat("SES-C4", "Actually I'm having chest pain right now")
    assert reply.reply == EMERGENCY_REPLY
    assert reply.tools_used == []
    conn = store.connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0] == 0
    finally:
        conn.close()
    # And the session still works afterwards for a routine turn.
    after = service.chat("SES-C4", "Okay, I need a cardiologist next week")
    assert after.reply != EMERGENCY_REPLY
    service.close()


def test_mild_headache_still_books_normally(db_path):
    service = ChatService(db_path=db_path, mock=True)
    reply = service.chat(
        "SES-C5", "I have a mild headache, need a doctor next week"
    )
    assert reply.reply != EMERGENCY_REPLY
    assert reply.reply != SELF_HARM_REPLY
    service.close()


def test_crisis_reply_kinds():
    assert crisis_reply("emergency") == EMERGENCY_REPLY
    assert crisis_reply("self_harm") == SELF_HARM_REPLY
    assert crisis_reply("anything-else") == EMERGENCY_REPLY
