"""Booking form: a deterministic slot-filling state machine.

The agent does NOT fill this form from the prompt. Deterministic code
tracks which slots are filled, asks for one missing piece at a time,
validates each answer as it arrives (Pydantic + the parsers below), and
applies corrections without restarting. The LLM only turns the machine's
questions and summaries into natural language.

Required slots, in the order they are asked:
    patient_ref, provider, slot (date + time), reason, contact_method
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from pydantic import BaseModel, Field

from .phi import mask_member_id, mask_name, mask_phone

WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def parse_natural_date(text: str, ref: date | None = None) -> date | None:
    """Parse a natural date deterministically. Returns None when unsure.

    Understands: today, tomorrow, "next Tuesday", "Tuesday", "Nov 17",
    "11/17", "2026-11-17", "next week" (Monday of next week). A bare
    weekday means the nearest upcoming occurrence; "next <weekday>" means
    the one a week later. Never guesses a year: month/day without a year
    means the next occurrence of that date (this year or next).
    """
    ref = ref or date.today()
    lowered = (text or "").lower().strip()
    if not lowered:
        return None
    if "today" in lowered:
        return ref
    if "tomorrow" in lowered:
        return ref + timedelta(days=1)
    if "next week" in lowered:
        days_ahead = (7 - ref.weekday()) % 7 or 7
        return ref + timedelta(days=days_ahead)

    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", lowered)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b", lowered)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        year = int(m.group(3)) if m.group(3) else ref.year
        if year < 100:
            year += 2000
        try:
            parsed = date(year, month, day)
        except ValueError:
            return None
        if parsed < ref:
            try:
                parsed = date(year + 1, month, day)
            except ValueError:
                return None
        return parsed
    m = re.search(
        r"\b(" + "|".join(MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", lowered
    )
    if m:
        month, day = MONTHS[m.group(1)], int(m.group(2))
        try:
            parsed = date(ref.year, month, day)
        except ValueError:
            return None
        if parsed < ref:
            parsed = date(ref.year + 1, month, day)
        return parsed
    m = re.search(
        r"\b(next\s+)?(" + "|".join(WEEKDAYS) + r")\b", lowered
    )
    if m:
        target = WEEKDAYS[m.group(2)]
        days_ahead = (target - ref.weekday()) % 7
        if m.group(1):
            days_ahead += 7
        return ref + timedelta(days=days_ahead)
    return None


def parse_time_of_day(text: str) -> str | None:
    """Return 'morning' or 'afternoon' when the user states a preference."""
    lowered = (text or "").lower()
    if "morning" in lowered:
        return "morning"
    if "afternoon" in lowered:
        return "afternoon"
    return None


def validate_phone(text: str) -> tuple[str | None, str | None]:
    """Normalize a US phone number to (XXX) XXX-XXXX, or return an error.

    Pulls the first phone-shaped run out of the text, so a sentence like
    'actually my phone is 612-555-0199' validates without other digits in
    the sentence polluting it.
    """
    match = re.search(
        r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text or ""
    )
    digits = re.sub(r"\D", "", match.group(0)) if match else ""
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None, (
            "That phone number doesn't look complete — I need 10 digits, "
            "like 612-555-0147."
        )
    return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}", None


def parse_provider_choice(text: str, providers: list[dict]) -> dict | None:
    """Pick a provider from the last shown list: 'second one', '#2', a name."""
    if not providers:
        return None
    lowered = (text or "").lower()
    ordinals = {
        "first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4,
        "1st": 0, "2nd": 1, "3rd": 2, "4th": 3, "5th": 4,
    }
    m = re.search(r"#\s*(\d+)", lowered)
    if m:
        idx = int(m.group(1)) - 1
        return providers[idx] if 0 <= idx < len(providers) else None
    for word, idx in ordinals.items():
        if word in lowered and idx < len(providers):
            return providers[idx]
    for provider in providers:
        last = provider.get("name", "").split()[-1].lower() if provider.get("name") else ""
        if last and last in lowered:
            return provider
    return None


def parse_slot_choice(text: str, slots: list[dict], ref: date | None = None) -> dict | None:
    """Pick a slot from the shown availability.

    Understands a date ("Wednesday"), a date + time of day ("Wednesday
    morning" -> first morning slot that day), or an exact time ("9:30").
    """
    if not slots:
        return None
    lowered = (text or "").lower()
    wanted_date = parse_natural_date(text, ref)
    tod = parse_time_of_day(text)
    m = re.search(r"\b(\d{1,2}):(\d{2})\s*(am|pm)?\b", lowered)
    wanted_time = None
    if m:
        hour, minute = int(m.group(1)), int(m.group(2))
        if m.group(3) == "pm" and hour < 12:
            hour += 12
        if m.group(3) == "am" and hour == 12:
            hour = 0
        wanted_time = f"{hour:02d}:{minute:02d}"
    candidates = list(slots)
    if wanted_date:
        candidates = [s for s in candidates if s.get("date") == wanted_date.isoformat()]
    if wanted_time:
        candidates = [s for s in candidates if s.get("time") == wanted_time]
    elif tod:
        if tod == "morning":
            candidates = [s for s in candidates if s.get("time", "") < "12:00"]
        else:
            candidates = [s for s in candidates if s.get("time", "") >= "12:00"]
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# The form itself
# ---------------------------------------------------------------------------


class BookingSummary(BaseModel):
    """Structured final summary. Pydantic, not free text — this is what the
    user confirms before anything is booked. All PII is masked."""

    patient_ref: str
    patient_masked: str
    phone_masked: str
    provider_name: str
    provider_npi: str
    slot: str
    date: str
    time: str
    reason: str
    contact_method: str
    missing: list[str] = Field(default_factory=list)

    def readable(self) -> str:
        lines = [
            "Here's what I have for your appointment:",
            f"- Patient: {self.patient_masked} ({self.patient_ref}), phone {self.phone_masked}",
            f"- Provider: {self.provider_name} (NPI {self.provider_npi})",
            f"- When: {self.date} at {self.time}",
            f"- Reason: {self.reason}",
            f"- Contact: {self.contact_method}",
        ]
        if self.missing:
            lines.append(f"Still needed: {', '.join(self.missing)}.")
        return "\n".join(lines)


@dataclass
class BookingForm:
    """One booking's slots. The state machine: every slot is either filled
    or missing, and the machine always knows which question comes next."""

    patient_ref: str = ""
    patient_masked: str = ""
    phone_masked: str = ""
    provider_npi: str = ""
    provider_name: str = ""
    slot: str = ""
    date: str = ""
    time: str = ""
    reason: str = ""
    contact_method: str = ""
    hold_id: str = ""
    appointment_id: str = ""
    stage: str = "collecting"  # collecting -> providers_shown -> slot_picked -> confirming -> booked
    # Context for resolving references like "the second one". Not slots —
    # just the last results shown, so follow-ups work without re-asking.
    last_providers: list = field(default_factory=list)
    last_slots: list = field(default_factory=list)

    def missing_slots(self) -> list[str]:
        missing = []
        if not self.patient_ref:
            missing.append("patient")
        if not self.provider_npi:
            missing.append("provider")
        if not self.slot:
            missing.append("slot")
        if not self.reason:
            missing.append("reason")
        if not self.contact_method:
            missing.append("contact_method")
        return missing

    def is_complete(self) -> bool:
        return not self.missing_slots()

    def next_question(self) -> str | None:
        """One missing piece at a time, in plain language."""
        missing = self.missing_slots()
        if not missing:
            return None
        first = missing[0]
        if first == "patient":
            return (
                "To book, I'll need your name, date of birth, and phone "
                "number. What's your full name?"
            )
        if first == "provider":
            return "Which provider would you like — the first one, the second, or someone else?"
        if first == "slot":
            return (
                "Which day works for you? A weekday name like 'Wednesday' "
                "or a date like 'Nov 18' is fine — and morning or afternoon?"
            )
        if first == "reason":
            return "What's the visit for, briefly?"
        return "How should the clinic reach you — a call or a text?"
        # contact_method

    def summary(self) -> BookingSummary:
        return BookingSummary(
            patient_ref=self.patient_ref,
            patient_masked=self.patient_masked,
            phone_masked=self.phone_masked,
            provider_name=self.provider_name,
            provider_npi=self.provider_npi,
            slot=self.slot,
            date=self.date,
            time=self.time,
            reason=self.reason,
            contact_method=self.contact_method,
            missing=self.missing_slots(),
        )


def apply_correction(form: BookingForm, text: str, ref: date | None = None) -> str | None:
    """Apply a mid-flow correction without restarting. Returns what changed.

    Understands: 'actually my phone is ...', 'make it Tuesday instead',
    'change the date to ...', 'actually it's for ...'. Returns None when
    the text carries no correction.
    """
    lowered = (text or "").lower()
    if "actually" not in lowered and "instead" not in lowered and "change" not in lowered:
        return None
    phone, _err = validate_phone(text)
    if phone and "phone" in lowered:
        form.phone_masked = mask_phone(text)
        return f"phone updated to {form.phone_masked}"
    new_date = parse_natural_date(text, ref)
    if new_date and ("tuesday" in lowered or "wednesday" in lowered or "thursday" in lowered
                     or "monday" in lowered or "friday" in lowered or "date" in lowered
                     or "instead" in lowered):
        form.date = new_date.isoformat()
        form.slot = ""
        form.time = ""
        form.hold_id = ""
        form.stage = "providers_shown"
        return f"date changed to {form.date}; please pick a time again"
    reason_match = re.search(r"actually (?:it'?s|it is) for (.+)$", lowered)
    if reason_match:
        form.reason = reason_match.group(1).strip().rstrip(".")
        return "reason updated"
    return None
