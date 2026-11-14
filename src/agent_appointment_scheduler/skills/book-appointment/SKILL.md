---
name: book-appointment
description: "Use when the user wants to book an appointment — collecting their details, picking a slot, and confirming the booking."
---

# Book an appointment

Booking is a slot-filling flow, and the slots are tracked by deterministic
code, not by the prompt. Your job is to ask for one missing piece at a
time and to call the tools in order. Never invent a slot, a time, or a
patient detail.

## The slots (in order)

1. Patient: full name, date of birth, phone. Call
   `create_patient_record` — it stores the details encrypted and returns
   only a patient reference (PAT-...) plus masked values. You only ever
   carry the reference and the masked values.
2. Provider: from the last `search_providers` result. "The second one"
   means index 2 of that list — use `choose_provider`.
3. Slot: date + time. Use `get_availability` for the provider, then
   `choose_slot`. A slot is held for 10 minutes while the user confirms;
   if the hold expires, pick the slot again.
4. Reason for visit: one short phrase, e.g. "heart checkup follow-up".
5. Contact method: text or call.

## Validation (deterministic, not your judgement)

- Dates: "next Tuesday", "Nov 18", "11/18" are parsed by code. If a
  date is rejected, repeat the exact error back and ask again.
- Phone: 10 digits. If it is rejected, ask for the full number again.
- Before confirming, call `get_booking_status` and show the masked
  summary to the user. Only after they say yes, call
  `book_appointment` with `confirmed=true` and an idempotency key.
  The same key twice returns the original appointment — never a
  duplicate.

## Corrections

- "Actually my phone is ..." or "make it Tuesday instead" changes just
  that slot. Do not restart the flow. Use `update_patient_contact` for
  phone changes and `choose_slot` again for a new day/time.
- If a tool call fails, call `get_plan` again and follow the new plan
  from the failed step. Do not improvise a different path.
