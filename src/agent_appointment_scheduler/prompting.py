"""Prompt layout and cache metrics.

Layout rule (cache-friendly, and the test proves it): the prompt is
built static-first, dynamic-last.

- Static prefix (byte-identical across every turn of every session):
  the base system prompt. The skill registry is injected right after it
  by the Strands AgentSkills plugin (stable text, same skills every
  turn), and the tool schemas are constant too.
- Dynamic content, appended after the prefix: the session id, the
  rolling summary, stored preferences, and the conversation history.
  Activated skill instructions arrive as tool results, also after the
  prefix.

Because the prefix never changes, the provider can cache it: OpenAI
caches automatically once the prompt is >= 1,024 tokens, and Gemini
caches implicitly (LiteLLM's table lists 1,024 tokens minimum for both;
Google's own docs note higher minimums, 2,048+, for some Gemini models
— check the current pricing page before budgeting on it). Cache reads
cost roughly 10% of normal input price on both providers. Below the
minimum the request is simply processed without caching — no error, no
behavior change. Caching is a cost/latency optimization, not a
correctness feature: the agent works identically with caching off.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import INPUT_PRICE_PER_MILLION

BASE_SYSTEM_PROMPT = """You are a healthcare appointment scheduling assistant. You help
people find providers and book, reschedule, or cancel appointments.

Rules you must follow:
- Use your tools to look things up. Never invent a provider, a slot, or a
  patient detail — call the tool and report what it returns.
- You explain results. You never decide clinical questions and never
  approve treatment. Booking decisions (slot free, confirmation, hold)
  belong to the deterministic policy layer, not to you.
- Before booking, show the masked summary and ask the user to confirm.
  Only call book_appointment with confirmed=true after they say yes.
- Patient details exist only as a reference (PAT-...) plus masked values.
  Never ask for or repeat a full name, date of birth, phone number, or
  member ID once a reference exists.
- Skills are listed in the available_skills section. When a skill applies
  to what the user is asking, call the skills tool to load its full
  instructions first, then follow them.
- If a tool call fails, call get_plan again and follow the replanned
  steps. Do not improvise a different path.
- All data is synthetic mock data for demonstration, except provider
  directory listings, which come from the public CMS NPI Registry.
"""


def static_prefix() -> str:
    """The cacheable static prefix: the base system prompt.

    Byte-identical across every turn of every session. The skill registry
    is injected after it by the Strands AgentSkills plugin (stable text,
    same skills every turn), and anything that changes per turn — session
    id, summary, preferences, history — goes after it via
    :func:`compose_system_prompt`.
    """
    return BASE_SYSTEM_PROMPT.rstrip()


def compose_system_prompt(
    session_id: str, summary: str = "", preferences: dict | None = None
) -> str:
    """Full system prompt for one turn: static prefix first, dynamic last.

    The session id is a dynamic block (it differs per session), so it
    sits after the static prefix — never inside it, or the provider's
    prefix cache would never hit. Same for the rolling summary and the
    stored preferences: they change as the conversation goes on, so they
    are appended after the static part.
    """
    parts = [static_prefix()]
    parts.append(
        f"\nSession ID: {session_id} — pass this as session_id to every "
        "stateful tool (search_providers, choose_provider, choose_slot, "
        "create_patient_record, get_booking_status, book_appointment, "
        "list_appointments, and the rest)."
    )
    if summary:
        parts.append(f"\nWhat you remember from earlier in this session:\n{summary}")
    if preferences:
        prefs = ", ".join(f"{key}={value}" for key, value in sorted(preferences.items()))
        parts.append(f"\nStored preferences for this user: {prefs}")
    return "\n".join(parts)


@dataclass
class CacheMetrics:
    """Per-turn cache accounting, from the provider's usage object."""

    provider: str
    model_id: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cache_hit_rate: float = 0.0
    est_cost_saved_usd: float = 0.0

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "prompt_tokens": self.prompt_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "est_cost_saved_usd": round(self.est_cost_saved_usd, 6),
        }


def extract_cache_metrics(
    provider: str, model_id: str, usage: dict | None
) -> CacheMetrics:
    """Pull cache numbers out of a model usage object.

    LiteLLM normalizes provider usage to the OpenAI shape:
    ``prompt_tokens_details.cached_tokens`` for OpenAI, and Gemini's
    ``usage_metadata.cachedContentTokenCount``. When the provider does
    not report cached tokens, they are 0 — the turn still worked, it
    just did not hit the cache. Cost saved assumes a cache read costs
    10% of a full-price input token.
    """
    usage = usage or {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(details.get("cached_tokens", 0) or 0)
    if not cached:
        cached = int(usage.get("cached_tokens", 0) or 0)
    if not cached:
        metadata = usage.get("usage_metadata") or {}
        cached = int(metadata.get("cachedContentTokenCount", 0) or 0)
    hit_rate = (cached / prompt_tokens) if prompt_tokens else 0.0
    price = INPUT_PRICE_PER_MILLION.get(model_id, 0.0)
    saved = cached * price * 0.9 / 1_000_000
    return CacheMetrics(
        provider=provider,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached,
        output_tokens=output_tokens,
        cache_hit_rate=hit_rate,
        est_cost_saved_usd=saved,
    )
