---
name: session-memory
description: "Use when the conversation is long or the user refers back to something from earlier — preferences, earlier bookings, or a previous session."
---

# Session memory

This agent remembers across turns and across sessions. Three layers:

- Working memory: the recent turns in context (bounded to 40 turns).
- Episodic memory: every turn is stored, and older turns are summarized
  into a rolling summary once history passes 12 turns. The summary is
  part of the context you get.
- Semantic memory: durable facts the user stated — prefers mornings,
  Minneapolis area, text contact. They are stored as preferences and
  recalled automatically. Use them: if the user prefers mornings, offer
  morning slots first without asking again.

## How to use it

- When the user says "the second one", "that one", "move it", or "cancel
  it", resolve the reference from the session history and the booking
  state — call `list_appointments` or `get_booking_status` to confirm
  what "it" is before acting.
- When the user returns in a later session, greet them with what you
  remember (masked): their preference and any upcoming appointment.
- If a preference conflicts with a new request ("mornings" but they ask
  for an afternoon slot), follow the new request and update the stored
  preference.
- Never repeat raw personal details from memory. Patient details exist
  only as a reference (PAT-...) and masked values.

## What not to do

- Do not claim to remember something that is not in the session state
  or the summary. If you are unsure, ask or look it up with a tool.
- Do not re-ask for information the session already has.
