"""Slot-filling tests: dates, phones, corrections, summary-then-confirm."""

from datetime import date, timedelta

import pytest

from agent_appointment_scheduler.slots import (
    BookingForm,
    apply_correction,
    parse_natural_date,
    parse_provider_choice,
    parse_slot_choice,
    validate_phone,
)

# Fixed reference: Saturday 2026-11-14 (the demo's "today").
REF = date(2026, 11, 14)


def test_parse_natural_dates():
    assert parse_natural_date("today", REF) == REF
    assert parse_natural_date("tomorrow", REF) == date(2026, 11, 15)
    assert parse_natural_date("Wednesday", REF) == date(2026, 11, 18)
    assert parse_natural_date("next Wednesday", REF) == date(2026, 11, 25)
    assert parse_natural_date("next Tuesday", REF) == date(2026, 11, 24)
    assert parse_natural_date("2026-11-18", REF) == date(2026, 11, 18)
    assert parse_natural_date("11/18", REF) == date(2026, 11, 18)
    assert parse_natural_date("Nov 17", REF) == date(2026, 11, 17)
    assert parse_natural_date("not a date", REF) is None


def test_parse_next_week():
    assert parse_natural_date("next week", REF) == date(2026, 11, 16)


def test_validate_phone():
    normalized, error = validate_phone("612-555-0147")
    assert normalized == "(612) 555-0147"
    assert error is None
    assert validate_phone("(612) 555-0147")[0] == "(612) 555-0147"
    assert validate_phone("6125550147")[0] == "(612) 555-0147"
    normalized, error = validate_phone("call me")
    assert normalized is None
    assert error and "10 digits" in error


def test_provider_choice():
    providers = [
        {"npi": "1", "name": "Amara Okafor"},
        {"npi": "2", "name": "Daniel Peterson"},
    ]
    assert parse_provider_choice("the second one", providers)["npi"] == "2"
    assert parse_provider_choice("#1", providers)["npi"] == "1"
    assert parse_provider_choice("Peterson", providers)["npi"] == "2"
    assert parse_provider_choice("nobody", providers) is None


def test_slot_choice():
    slots = [
        {"slot": "2026-11-18T09:00", "date": "2026-11-18", "time": "09:00", "available": True},
        {"slot": "2026-11-18T13:00", "date": "2026-11-18", "time": "13:00", "available": True},
        {"slot": "2026-11-19T09:00", "date": "2026-11-19", "time": "09:00", "available": True},
    ]
    assert parse_slot_choice("Wednesday morning", slots, REF)["slot"] == "2026-11-18T09:00"
    assert parse_slot_choice("2026-11-19T09:00", slots, REF)["slot"] == "2026-11-19T09:00"
    assert parse_slot_choice("Friday", slots, REF) is None


def test_form_missing_and_next_question():
    form = BookingForm()
    assert form.missing_slots() == ["patient", "provider", "slot", "reason", "contact_method"]
    assert "name" in (form.next_question() or "")
    form.patient_ref = "PAT-1"
    form.provider_npi = "1215596358"
    form.provider_name = "Dr Abbasi"
    form.slot = "2026-11-18T09:00"
    form.date = "2026-11-18"
    form.time = "09:00"
    assert form.missing_slots() == ["reason", "contact_method"]
    form.reason = "checkup"
    form.contact_method = "text"
    assert form.missing_slots() == []
    assert form.is_complete()
    assert form.next_question() is None


def test_summary_is_masked():
    form = BookingForm()
    form.patient_ref = "PAT-1"
    form.patient_masked = "J*** L***"
    form.phone_masked = "***-***-0147"
    form.provider_name = "Dr Abbasi"
    form.slot = "2026-11-18T09:00"
    form.reason = "checkup"
    form.contact_method = "text"
    summary = form.summary()
    assert summary.patient_ref == "PAT-1"
    assert summary.phone_masked == "***-***-0147"
    assert "612-555-0147" not in summary.readable()
    assert "Jordan" not in summary.readable()


def test_apply_correction_phone():
    form = BookingForm()
    form.phone_masked = "***-***-0147"
    changed = apply_correction(form, "actually my phone is 612-555-0199")
    assert changed and "0199" in changed
    assert form.phone_masked == "***-***-0199"


def test_apply_correction_day():
    form = BookingForm()
    form.slot = "2026-11-18T09:00"
    form.date = "2026-11-18"
    form.time = "09:00"
    form.hold_id = "H-1"
    changed = apply_correction(form, "make it Thursday instead", REF)
    assert changed and "2026-11-19" in changed
    assert form.date == "2026-11-19"
    assert form.slot == ""  # the old slot is released; pick a time again
    assert form.stage == "providers_shown"


def test_apply_correction_no_correction():
    form = BookingForm()
    assert apply_correction(form, "hello there") is None
