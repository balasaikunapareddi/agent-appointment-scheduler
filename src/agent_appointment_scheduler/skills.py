"""Skill-based prompts (progressive disclosure).

The agent's know-how lives in ``skills/*/SKILL.md``, one file per skill,
each with frontmatter (name + WHEN it applies) and instructions (HOW).
The agent always carries only the registry — names plus one-line
descriptions — and a skill's full instructions load only when the
agent activates it, via the native Strands ``AgentSkills`` plugin and
its ``skills`` tool.

This module is the deterministic side of that pattern:
- ``load_skills`` reads the SKILL.md files through the native ``Skill``
  dataclass (so the registry text matches what the plugin injects).
- ``registry_text`` renders the stable registry block.
- ``select_skills`` picks skills from the user's message + booking stage
  with keyword rules. The scripted model and the tests use this; a real
  model uses the same keywords as guidance and the ``skills`` tool to
  activate. Either way, only relevant skills load — the base system
  prompt stays small.
"""

from __future__ import annotations

import re
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

_SKILL_KEYWORDS: dict[str, list[str]] = {
    "find-providers": [
        "find", "search", "looking for", "need a", "cardiologist",
        "doctor", "provider", "clinic", "specialist", "family medicine",
        "internist", "ob/gyn", "near",
    ],
    "book-appointment": [
        "book", "schedule", "appointment", "slot", "morning", "afternoon",
        "tuesday", "wednesday", "thursday", "monday", "friday",
        "my name is", "born", "phone", "confirm",
    ],
    "reschedule-cancel": [
        "reschedule", "move it", "change it", "cancel", "different day",
        "instead", "move my appointment",
    ],
    "session-memory": [
        "second one", "that one", "the one", "earlier", "last time",
        "my usual", "remember", "what did i book", "my appointment",
    ],
}


def load_skills(skills_dir: Path | None = None):
    """Load every skill in the directory via the native Skill dataclass."""
    from strands.vended_plugins.skills import Skill

    directory = Path(skills_dir or SKILLS_DIR)
    skills = []
    for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
        md = sub / "SKILL.md"
        if md.is_file():
            # from_file loads one skill directory; from_directory is for a
            # parent directory of skills and would return [] here.
            skills.append(Skill.from_file(sub))
    return skills


def registry_text(skills: list | None = None) -> str:
    """The stable registry block: names + one-line descriptions.

    This is the text the agent carries on every turn. Full instructions
    stay out until a skill is activated.
    """
    loaded = skills if skills is not None else load_skills()
    lines = ["Available skills (call the skills tool to load one):"]
    for skill in loaded:
        lines.append(f"- {skill.name}: {skill.description}")
    return "\n".join(lines)


def select_skills(text: str, stage: str = "collecting") -> list[str]:
    """Deterministically pick the skills a turn needs.

    Keyword rules over the user message, plus the booking stage: once a
    booking exists, reschedule/cancel is always in scope; once history
    is long, session-memory is too. Returns skill names in registry
    order, never duplicates.
    """
    lowered = (text or "").lower()
    picked: list[str] = []
    for name, keywords in _SKILL_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            picked.append(name)
    if stage in ("booked", "confirming") and "reschedule-cancel" not in picked:
        picked.append("reschedule-cancel")
    order = ["find-providers", "book-appointment", "reschedule-cancel", "session-memory"]
    return [name for name in order if name in picked]


def skill_instructions(name: str) -> str:
    """Full instructions of one skill, for tests and the scripted model."""
    for skill in load_skills():
        if skill.name == name:
            return skill.instructions
    raise KeyError(f"Unknown skill: {name}")
