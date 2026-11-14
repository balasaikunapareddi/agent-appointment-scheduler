"""Scripted model for offline tests and the mock demo.

A deterministic stand-in for a real LLM. It plans the booking flow from
the conversation history alone (no model call, no key) and calls the
same MCP tools a real model would: search -> choose provider -> slots ->
hold -> patient details -> summary -> confirm -> book -> reschedule.

It also activates skills the same way a real model does: on a turn whose
intent matches a skill (see ``skills.select_skills``), it calls the
``skills`` tool first, then proceeds with the domain tool. That is how
the tests prove progressive disclosure: a provider-search turn loads
only find-providers, a booking turn loads the booking skill.

Session state is reconstructed from the message history on every turn
(tool results carry providers, slots, the patient reference, and the
appointment id), so a fresh agent instance resuming a session picks up
exactly where the old one left off.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterable
from datetime import datetime, timedelta
from typing import Any

from strands.models import Model
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from .skills import select_skills
from .slots import parse_natural_date


def _result_json(message: dict) -> dict | None:
    """Pull the JSON payload out of a toolResult content block."""
    for block in message.get("content", []):
        result = block.get("toolResult")
        if not result:
            continue
        for item in result.get("content", []):
            if "json" in item:
                return item["json"] if isinstance(item["json"], dict) else {"value": item["json"]}
            if "text" in item:
                try:
                    parsed = json.loads(item["text"])
                    return parsed if isinstance(parsed, dict) else {"value": parsed}
                except (json.JSONDecodeError, TypeError):
                    return {"text": item["text"]}
    return None


def _tool_use_name(message: dict) -> str | None:
    for block in message.get("content", []):
        use = block.get("toolUse")
        if use:
            return use.get("name")
    return None


def _user_text(message: dict) -> str:
    parts = []
    for block in message.get("content", []):
        if "text" in block:
            parts.append(block["text"])
    return " ".join(parts)


def _next_week_range() -> tuple[str, str]:
    today = datetime.now().date()
    days_to_monday = (7 - today.weekday()) % 7 or 7
    monday = today + timedelta(days=days_to_monday)
    return monday.isoformat(), (monday + timedelta(days=4)).isoformat()


class ScriptedModel(Model):
    """Deterministic booking-flow model. No network, no key, no cost."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {"model_id": "scripted-mock"}
        self._pending: dict[str, str] = {}
        self._tool_use_count = 0

    def get_config(self) -> Any:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError("ScriptedModel does not support structured output.")

    # -- state reconstruction -------------------------------------------------

    def _state(self, messages: Messages) -> dict:
        state: dict[str, Any] = {
            "providers": [],
            "provider": None,
            "slots": [],
            "slot": "",
            "patient_ref": "",
            "appointment_id": "",
            "status_shown": False,
            "skills_activated_this_turn": [],
            "last_tool": "",
            "last_result": {},
        }
        last_user_index = -1
        for i, message in enumerate(messages):
            if message.get("role") == "user" and _user_text(message):
                last_user_index = i
        for i, message in enumerate(messages):
            name = _tool_use_name(message)
            if name:
                state["last_tool"] = name
                if i > last_user_index and name == "skills":
                    state["skills_activated_this_turn"].append(message)
                continue
            result = _result_json(message)
            if result is None:
                continue
            state["last_result"] = result
            tool = state["last_tool"]
            if tool == "search_providers" and result.get("ok"):
                state["providers"] = result.get("providers", [])
            elif tool == "get_availability" and result.get("ok"):
                state["slots"] = result.get("slots", [])
            elif tool == "choose_provider" and result.get("ok"):
                state["provider"] = result.get("provider")
            elif tool == "choose_slot" and result.get("ok"):
                state["slot"] = result.get("slot", "")
            elif tool == "create_patient_record" and result.get("ok"):
                state["patient_ref"] = result.get("patient_ref", "")
            elif tool == "book_appointment" and result.get("ok"):
                state["appointment_id"] = result.get("appointment_id", "")
            elif tool == "get_booking_status" and result.get("ok"):
                state["status_shown"] = True
            elif tool == "reschedule_appointment" and result.get("ok"):
                state["slot"] = result.get("slot", state["slot"])
        return state

    # -- decision ---------------------------------------------------------------

    def _decide_tool_call(
        self, messages: Messages, tool_specs: list[ToolSpec] | None
    ) -> tuple[str, dict] | None:
        state = self._state(messages)
        available = {spec.get("name") for spec in (tool_specs or [])}
        last = messages[-1] if messages else {}
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user" and _user_text(message):
                last_user = _user_text(message)
                break
        lowered = last_user.lower()
        session_id = self._session_id(messages)

        def can(name: str) -> bool:
            return name in available

        # A tool result just came back: chain the next step.
        if _tool_use_name(last) is None and _result_json(last) is not None:
            if state["last_tool"] == "skills":
                # Skill loaded — now act on the user's intent, not end the turn.
                return self._from_user(state, lowered, last_user, session_id, can)
            return self._after_tool(state, lowered, session_id, can)

        # A fresh user message: activate the skill for this intent first.
        if can("skills") and not state["skills_activated_this_turn"]:
            wanted = select_skills(last_user, self._stage(state))
            if wanted:
                return ("skills", {"skill_name": wanted[0]})

        return self._from_user(state, lowered, last_user, session_id, can)

    def _after_tool(self, state: dict, lowered: str, session_id: str, can) -> tuple[str, dict] | None:
        tool = state["last_tool"]
        result = state["last_result"]
        if not result.get("ok"):
            return None
        if tool == "choose_provider" and can("get_availability"):
            provider = state["provider"] or {}
            date_from, date_to = _next_week_range()
            return ("get_availability", {
                "npi": provider.get("npi", ""),
                "date_from": date_from,
                "date_to": date_to,
                "time_of_day": "morning" if "morning" in lowered else "",
            })
        if tool == "get_availability" and ("reschedule" in lowered or "move" in lowered) and can("reschedule_appointment"):
            target_date = self._target_date(lowered)
            slot = self._pick_slot(state["slots"], target_date, lowered)
            if slot and state["appointment_id"]:
                return ("reschedule_appointment", {
                    "appointment_id": state["appointment_id"],
                    "new_slot": slot,
                    "session_id": session_id,
                })
            return None
        if tool == "get_availability" and can("choose_slot") and state["slots"]:
            # Only hold a slot when the user already named a day (e.g. the
            # reschedule turn, or "Wednesday morning" after availability
            # was shown). Otherwise present the slots and let them pick —
            # auto-holding the first slot of the week is presumptuous.
            target_date = self._target_date(lowered)
            if target_date is None:
                return None
            slot = self._pick_slot(state["slots"], target_date, lowered)
            if slot:
                return ("choose_slot", {"slot": slot, "session_id": session_id})
            return None
        if tool == "set_visit_details" and can("get_booking_status"):
            return ("get_booking_status", {"session_id": session_id})
        if tool == "update_patient_contact" and can("get_booking_status"):
            return ("get_booking_status", {"session_id": session_id})
        if tool == "list_appointments":
            appointments = result.get("appointments", [])
            if not appointments:
                return None
            appointment = appointments[0]
            state["appointment_id"] = appointment.get("appointment_id", "")
            if "cancel" in lowered and can("cancel_appointment"):
                return ("cancel_appointment", {
                    "appointment_id": appointment["appointment_id"],
                    "session_id": session_id,
                })
            if ("reschedule" in lowered or "move" in lowered) and can("get_availability"):
                target_date = self._target_date(lowered)
                date = target_date or _next_week_range()[0]
                return ("get_availability", {
                    "npi": appointment.get("npi", ""),
                    "date_from": date,
                    "date_to": date,
                    "time_of_day": "morning" if "morning" in lowered else "",
                })
            return None
        return None

    def _from_user(self, state: dict, lowered: str, last_user: str, session_id: str, can) -> tuple[str, dict] | None:
        if ("cancel" in lowered and "appointment" in lowered) or lowered.strip().startswith("cancel"):
            if can("list_appointments"):
                return ("list_appointments", {"session_id": session_id, "patient_ref": state["patient_ref"]})
            return None
        if "reschedule" in lowered or "move it" in lowered or "move my appointment" in lowered:
            if state["appointment_id"] and can("get_availability"):
                target_date = self._target_date(lowered)
                date = target_date or _next_week_range()[0]
                npi = (state["provider"] or {}).get("npi", "")
                if not npi and state["slots"]:
                    npi = ""
                return ("get_availability", {
                    "npi": npi,
                    "date_from": date,
                    "date_to": date,
                    "time_of_day": "morning" if "morning" in lowered else "",
                }) if npi else ("list_appointments", {"session_id": session_id, "patient_ref": state["patient_ref"]})
            if can("list_appointments"):
                return ("list_appointments", {"session_id": session_id, "patient_ref": state["patient_ref"]})
            return None
        if any(phrase in lowered for phrase in ("what's my appointment", "what is my appointment", "my appointment", "when is my")):
            if can("list_appointments"):
                return ("list_appointments", {"session_id": session_id, "patient_ref": state["patient_ref"]})
            return None
        if lowered.strip() in ("yes", "yes please", "confirm", "book it", "looks good", "go ahead", "yes, book it"):
            if state["patient_ref"] and state["provider"] and state["slot"] and state["status_shown"]:
                if can("book_appointment"):
                    return ("book_appointment", {
                        "session_id": session_id,
                        "confirmed": True,
                        "idempotency_key": f"{session_id}-{state['slot']}",
                    })
            if can("get_booking_status"):
                return ("get_booking_status", {"session_id": session_id})
            return None
        choice = self._provider_choice(lowered, state["providers"])
        if choice is not None and can("choose_provider"):
            return ("choose_provider", {"choice": choice, "session_id": session_id})
        details = self._extract_patient_details(last_user)
        if details and can("create_patient_record") and not state["patient_ref"]:
            self._pending.update(details)
            if all(k in self._pending for k in ("full_name", "dob", "phone")):
                return ("create_patient_record", {
                    "full_name": self._pending["full_name"],
                    "dob": self._pending["dob"],
                    "phone": self._pending["phone"],
                    "member_id": self._pending.get("member_id", ""),
                    "contact_method": "text",
                    "session_id": session_id,
                })
            return None
        if details and state["patient_ref"] and "phone" in details and can("update_patient_contact"):
            return ("update_patient_contact", {
                "patient_ref": state["patient_ref"],
                "phone": details["phone"],
                "session_id": session_id,
            })
        if ("heart checkup" in lowered or "checkup" in lowered or "follow-up" in lowered or "follow up" in lowered) and can("set_visit_details"):
            reason = "heart checkup follow-up" if "heart" in lowered else last_user.strip()[:200]
            return ("set_visit_details", {
                "reason": reason,
                "contact_method": "text" if "text" in lowered else "",
                "session_id": session_id,
            })
        if state["provider"] and state["slots"]:
            target_date = self._target_date(lowered)
            slot = self._pick_slot(state["slots"], target_date, lowered)
            if slot and can("choose_slot"):
                return ("choose_slot", {"slot": slot, "session_id": session_id})
        if any(word in lowered for word in ("cardiologist", "doctor", "provider", "find", "need a", "looking for")) and can("search_providers"):
            specialty = "cardiology" if "cardio" in lowered else "family medicine"
            return ("search_providers", {
                "specialty": specialty,
                "city": "Minneapolis",
                "state": "MN",
                "session_id": session_id,
            })
        if state["provider"] and not state["slots"] and can("get_availability"):
            date_from, date_to = _next_week_range()
            return ("get_availability", {
                "npi": state["provider"].get("npi", ""),
                "date_from": date_from,
                "date_to": date_to,
                "time_of_day": "morning" if "morning" in lowered else "",
            })
        return None

    # -- helpers -----------------------------------------------------------------

    def _session_id(self, messages: Messages) -> str:
        return getattr(self, "_system_session_id", "")

    def _stage(self, state: dict) -> str:
        if state["appointment_id"]:
            return "booked"
        if state["slot"]:
            return "slot_picked"
        if state["providers"]:
            return "providers_shown"
        return "collecting"

    @staticmethod
    def _target_date(lowered: str) -> str | None:
        for word in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
            if word in lowered:
                parsed = parse_natural_date(word)
                if parsed:
                    return parsed.isoformat()
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", lowered)
        return match.group(1) if match else None

    @staticmethod
    def _pick_slot(slots: list[dict], target_date: str | None, lowered: str) -> str | None:
        candidates = [s for s in slots if s.get("available")]
        if target_date:
            candidates = [s for s in candidates if s["slot"].startswith(target_date)]
        if "morning" in lowered:
            candidates = [s for s in candidates if s["slot"][11:13] < "12"]
        if "afternoon" in lowered:
            candidates = [s for s in candidates if s["slot"][11:13] >= "12"]
        return candidates[0]["slot"] if candidates else None

    @staticmethod
    def _provider_choice(lowered: str, providers: list[dict]) -> str | None:
        if not providers:
            return None
        match = re.search(r"\b(second|2nd|#2)\b", lowered)
        if match and len(providers) >= 2:
            return "second one"
        match = re.search(r"\b(first|1st|#1)\b", lowered)
        if match:
            return "first one"
        for provider in providers:
            name = provider.get("name", "").lower()
            if name and name.split()[-1] in lowered:
                return provider["name"]
        return None

    @staticmethod
    def _extract_patient_details(text: str) -> dict:
        details: dict[str, str] = {}
        name = re.search(r"my name is ([A-Za-z]+ [A-Za-z]+)", text, re.IGNORECASE)
        if name:
            details["full_name"] = name.group(1).strip()
        dob = re.search(r"born ([A-Za-z]+ \d{1,2}, \d{4})", text, re.IGNORECASE)
        if dob:
            try:
                details["dob"] = datetime.strptime(dob.group(1), "%B %d, %Y").date().isoformat()
            except ValueError:
                pass
        dob_iso = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)
        if dob_iso:
            details["dob"] = dob_iso.group(1)
        phone = re.search(r"(\d{3}[-.\s]\d{3}[-.\s]\d{4})", text)
        if phone:
            details["phone"] = phone.group(1)
        member = re.search(r"member (?:id )?([A-Za-z0-9-]+)", text, re.IGNORECASE)
        if member:
            details["member_id"] = member.group(1)
        return details

    def _final_text(self, messages: Messages, state: dict) -> str:
        tool = state["last_tool"]
        result = state["last_result"]
        if tool == "search_providers" and result.get("ok"):
            providers = result.get("providers", [])
            lines = [f"I found {len(providers)} cardiologists near Minneapolis:"]
            for i, provider in enumerate(providers, 1):
                lines.append(
                    f"{i}. {provider['name']} — {provider.get('specialty', '')}, "
                    f"{provider.get('organization', '')}, {provider.get('city', '')} "
                    f"({provider.get('phone', '')})"
                )
            if result.get("source") == "seed":
                lines.append("(These come from the local seed list — the CMS registry was unreachable.)")
            lines.append("Which one would you like — the first or the second?")
            return "\n".join(lines)
        if tool == "get_availability" and result.get("ok"):
            slots = [s for s in result.get("slots", []) if s.get("available")]
            provider = state["provider"] or {}
            if not slots:
                return "I couldn't find a free slot in that window. Want to try a different day?"
            lines = [f"Here are the open slots for {provider.get('name', 'this provider')} next week:"]
            for slot in slots[:6]:
                lines.append(f"- {slot['slot']}")
            lines.append("Which day and time works for you?")
            return "\n".join(lines)
        if tool == "choose_slot" and result.get("ok"):
            return (
                f"I've held {result['slot']} for 10 minutes while we finish. "
                "To complete the booking I need your full name, date of birth, and phone number."
            )
        if tool == "create_patient_record" and result.get("ok"):
            return (
                f"Thanks — I've saved your details under patient reference {result['patient_ref']} "
                f"(name {result['name_masked']}, phone {result['phone_masked']}). "
                "What's the reason for the visit, and should the clinic reach you by text or call?"
            )
        if tool == "get_booking_status" and result.get("ok"):
            return result.get("readable_summary", "") + "\nDoes that look right? Say yes and I'll book it."
        if tool == "book_appointment" and result.get("ok"):
            return (
                f"Booked! Your appointment is {result['appointment_id']} with "
                f"{result.get('provider_name', '')} on {result.get('slot', '')}. "
                f"Reason: {result.get('reason', '')}. The clinic will reach you by "
                f"{result.get('contact_method', 'text')}."
            )
        if tool == "list_appointments" and result.get("ok"):
            appointments = result.get("appointments", [])
            if not appointments:
                return "I don't see any appointment on this session yet."
            appointment = appointments[0]
            return (
                f"You have an appointment {appointment['appointment_id']} with "
                f"{appointment.get('provider_name', '')} on {appointment.get('slot', '')} "
                f"({appointment.get('status', '')})."
            )
        if tool == "reschedule_appointment" and result.get("ok"):
            return (
                f"Moved it — your appointment {result['appointment_id']} is now on "
                f"{result['slot']} with {result.get('provider_name', '')}. "
                "The old slot is free again."
            )
        if tool == "cancel_appointment" and result.get("ok"):
            return f"Cancelled {result['appointment_id']}. Let me know if you want to book a new time."
        if tool == "update_patient_contact" and result.get("ok"):
            return f"Updated — your phone on file is now {result['phone_masked']}."
        if tool == "skills":
            return "Let me look that up."
        if not result.get("ok", True) and result.get("error"):
            return f"That didn't work: {result['error']}"
        last_user = ""
        for message in reversed(messages):
            if message.get("role") == "user" and _user_text(message):
                last_user = _user_text(message)
                break
        if "prefer" in last_user.lower() and "morning" in last_user.lower():
            return "Got it — mornings, Minneapolis area. I'll offer morning slots first."
        return "How can I help — find a provider, book an appointment, or change an existing one?"

    # -- streaming ---------------------------------------------------------------

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        # The session id travels in the system prompt as a dynamic block
        # (after the static prefix, so caching is unaffected).
        self._system_session_id = ""
        if system_prompt:
            match = re.search(r"Session ID:\s*(\S+)", system_prompt)
            if match:
                self._system_session_id = match.group(1)
        call = self._decide_tool_call(messages, tool_specs)
        self._tool_use_count += 1
        if call is not None:
            name, args = call
            tool_use_id = f"scripted-{self._tool_use_count:04d}"
            yield {"messageStart": {"role": "assistant"}}
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}}
                }
            }
            yield {
                "contentBlockDelta": {
                    "delta": {"toolUse": {"input": json.dumps(args)}}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}}
            return
        state = self._state(messages)
        text = self._final_text(messages, state)
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {"usage": {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}, "metrics": {"latencyMs": 0}}}
