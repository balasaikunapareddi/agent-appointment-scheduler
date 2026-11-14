---
name: reschedule-cancel
description: "Use when the user wants to change the time of, or cancel, an existing appointment."
---

# Reschedule or cancel

The user already has a booking. Find it first, then change it — never
book a second appointment to "replace" one.

## How to reschedule

- Call `list_appointments` for the session to find the booking (you
  only need the appointment id, e.g. APT-1001).
- Ask which day and time they want instead, or read it from what they
  just said ("move it to Thursday morning").
- Check `get_availability` for the provider on the new day, pick a free
  slot, then call `reschedule_appointment`. The old slot is freed
  automatically and the appointment id stays the same.
- If the new slot is already booked, say so and offer the nearest free
  one. Do not silently pick a different day.

## How to cancel

- Call `cancel_appointment` with the appointment id. It refuses an
  already-cancelled appointment — report that instead of trying again.
- After cancelling, confirm the cancellation and offer to book a new
  time if the user wants one.

## What not to do

- Do not re-ask for the patient's name, date of birth, or phone: the
  booking already has a patient reference.
- Do not change the provider during a reschedule. If the user wants a
  different provider, that is a new booking.
