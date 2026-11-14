"""Deterministic crisis screen: emergencies never reach the model.

The agent books routine appointments. It must never be the thing that
decides whether "I'm having chest pain right now" is urgent, and it must
never answer an emergency with a routine slot next week. So every turn
passes through this screen *before* any agent or model is created:

- :func:`detect_crisis` matches a deliberately narrow list of emergency
  phrases (word-boundary regex over normalized text). A bare "pain" does
  not trigger; "chest pain" does. The list errs toward caution: a
  historical mention ("I had a seizure last year") will trigger too,
  because the cost of the fixed reply is a wasted 911 suggestion while
  the cost of missing a real emergency is a missed emergency.
- The reply is fixed text, not model output: the model can neither
  soften it nor improvise a different path, and no tool is ever called
  on a crisis turn (no booking, no hold, no provider search).
- Two replies: a medical emergency (call 911 / nearest emergency
  department) and thoughts of self-harm (988 Suicide and Crisis
  Lifeline, plus 911 if they might act on it).

This is the same principle as the rest of the agent — decisions live in
deterministic code, not in the prompt — with the system prompt carrying
one backup line in case the screen is ever bypassed.
"""

from __future__ import annotations

import re

EMERGENCY_REPLY = (
    "This sounds like an emergency — call 911 now, or go to your nearest "
    "emergency department immediately. Don't wait for an appointment; "
    "I can't book emergency care. If you can't call yourself, ask someone "
    "nearby to call for you."
)

SELF_HARM_REPLY = (
    "I'm really concerned about you. Please call or text 988 right now — "
    "that's the Suicide and Crisis Lifeline, answered 24/7 in the US. "
    "If you feel you might act on these thoughts, call 911 or go to your "
    "nearest emergency department immediately."
)

# Deliberately narrow: specific symptoms and events, never a bare word
# like "pain" or "hurt" on its own.
_EMERGENCY_PHRASES = (
    "chest pain",
    "chest hurts",
    "chest hurt",
    "chest pressure",
    "chest tightness",
    "can't breathe",
    "cannot breathe",
    "cant breathe",
    "trouble breathing",
    "difficulty breathing",
    "not breathing",
    "stopped breathing",
    "isn't breathing",
    "isnt breathing",
    "can't catch my breath",
    "short of breath",
    "choking",
    "bleeding heavily",
    "heavy bleeding",
    "bleeding a lot",
    "won't stop bleeding",
    "wont stop bleeding",
    "blood everywhere",
    "unconscious",
    "passed out",
    "fainted",
    "blacked out",
    "not waking up",
    "won't wake up",
    "wont wake up",
    "having a stroke",
    "signs of a stroke",
    "face drooping",
    "slurred speech",
    "sudden numbness",
    "severe allergic reaction",
    "anaphylaxis",
    "throat closing",
    "throat swelling",
    "overdose",
    "took too many pills",
    "took too much medicine",
    "seizure",
    "having a seizure",
    "convulsion",
    "been shot",
    "was shot",
    "stabbed",
)

_SELF_HARM_PHRASES = (
    "kill myself",
    "killing myself",
    "end my life",
    "take my own life",
    "hurt myself",
    "harm myself",
    "self-harm",
    "self harm",
    "suicidal",
    "suicide",
    "don't want to live",
    "dont want to live",
    "want to die",
)


def _compile(phrases: tuple[str, ...]) -> list[re.Pattern]:
    """One word-boundary regex per phrase, over normalized text."""
    return [
        re.compile(r"\b" + re.escape(phrase) + r"\b") for phrase in phrases
    ]


_EMERGENCY_PATTERNS = _compile(_EMERGENCY_PHRASES)
_SELF_HARM_PATTERNS = _compile(_SELF_HARM_PHRASES)


def _normalize(message: str) -> str:
    text = message.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def detect_crisis(message: str) -> str | None:
    """Return ``"emergency"`` or ``"self_harm"`` for a crisis message.

    Returns ``None`` for anything else — including ordinary complaints
    ("mild headache", "pain in my knee") and routine booking requests.
    A message matching both kinds is an emergency: 911 is the right
    answer for an overdose either way.
    """
    text = _normalize(message)
    if any(p.search(text) for p in _EMERGENCY_PATTERNS):
        return "emergency"
    if any(p.search(text) for p in _SELF_HARM_PATTERNS):
        return "self_harm"
    return None


def crisis_reply(kind: str) -> str:
    """The fixed reply for a detected crisis kind."""
    if kind == "self_harm":
        return SELF_HARM_REPLY
    return EMERGENCY_REPLY
