"""PHI handling: masking and redaction.

The rule this module enforces: raw sensitive values (full name, date of
birth, phone, member/insurance ID) never travel in model context, logs, or
stored conversation history. Code stores them encrypted (see ``crypto``)
and hands the model only an opaque patient reference plus masked values.

Masking conventions used everywhere in this repo:
- phone: ``***-***-4674`` (last 4 digits only)
- date of birth: ``**/**/1990`` (year only)
- member ID: ``***-***-1234`` (last 4 characters)
- name: first initial + masked rest, e.g. ``J*** L***``
"""

from __future__ import annotations

import re

_PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
_DOB_RE = re.compile(
    r"\b\d{2}/\d{2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b"
    r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2},\s*\d{4}\b"
)
_MEMBER_RE = re.compile(r"\bMBR-[A-Za-z0-9]+\b", re.IGNORECASE)
_PATIENT_REF_RE = re.compile(r"\bPAT-[A-Za-z0-9]+\b")


def mask_phone(phone: str) -> str:
    """Keep only the last 4 digits of a phone number."""
    digits = re.sub(r"\D", "", phone or "")
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return "***-***-****"
    return f"***-***-{digits[-4:]}"


def mask_dob(dob: str) -> str:
    """Keep only the year of a date of birth.

    Accepts ``YYYY-MM-DD`` or ``MM/DD/YYYY``; anything else is fully masked.
    """
    dob = (dob or "").strip()
    m = re.match(r"^(\d{4})-\d{2}-\d{2}$", dob)
    if m:
        return f"**/**/{m.group(1)}"
    m = re.match(r"^\d{2}/\d{2}/(\d{4})$", dob)
    if m:
        return f"**/**/{m.group(1)}"
    return "**/**/****"


def mask_member_id(member_id: str) -> str:
    """Keep only the last 4 characters of a member/insurance ID."""
    value = (member_id or "").strip()
    if len(value) <= 4:
        return "***-***-****"
    return f"***-***-{value[-4:]}"


def mask_name(full_name: str) -> str:
    """First initial of each name part, the rest masked: ``J*** L***``."""
    parts = (full_name or "").split()
    if not parts:
        return "***"
    return " ".join(f"{p[0].upper()}***" for p in parts if p)


def mask_patient_record(record: dict) -> dict:
    """Return the masked, model-safe view of a patient record dict."""
    return {
        "patient_ref": record.get("patient_ref"),
        "name_masked": mask_name(record.get("full_name", "")),
        "dob_masked": mask_dob(record.get("dob", "")),
        "phone_masked": mask_phone(record.get("phone", "")),
        "member_id_masked": mask_member_id(record.get("member_id", ""))
        if record.get("member_id")
        else None,
        "contact_method": record.get("contact_method"),
    }


class PIIRedactor:
    """Redact sensitive values out of arbitrary text.

    Regexes cover phones, dates of birth and member IDs. Literal values
    (e.g. a patient's actual name) can be registered per session so they
    are masked wherever they appear in logs, traces or stored history.
    """

    def __init__(self, literals: list[str] | None = None) -> None:
        self.literals = sorted(
            {v for v in (literals or []) if v and len(v.strip()) >= 3},
            key=len,
            reverse=True,
        )

    def register(self, *values: str) -> None:
        for value in values:
            if value and len(value.strip()) >= 3 and value not in self.literals:
                self.literals.append(value)
        self.literals.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        if not text:
            return text
        out = str(text)
        for literal in self.literals:
            out = out.replace(literal, "[redacted]")
        out = _PHONE_RE.sub("***-***-****", out)
        out = _DOB_RE.sub("**/**/****", out)
        out = _MEMBER_RE.sub("MBR-****", out)
        return out


def redact_text(text: str, literals: list[str] | None = None) -> str:
    """One-shot redaction without keeping a redactor around."""
    return PIIRedactor(literals).redact(text)
