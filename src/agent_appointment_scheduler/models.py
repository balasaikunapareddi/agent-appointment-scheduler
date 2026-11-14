"""Model providers and fallback.

One real model per provider, chosen the same way as in
agent-eligibility-claims (``PROVIDERS`` table, key from env or ``.env``,
fail fast with a clear message — never a silent fallback to mock).

Fallbacks: ``ModelWithFallback`` wraps two Strands models. It streams
from the primary; if the primary fails BEFORE any content was produced
(rate limit, auth, outage), it retries the same turn on the fallback
model. If content already streamed, the turn cannot be restarted, so the
error propagates. This is how a Gemini 429 (free-tier per-minute limit)
falls back to OpenAI, and vice versa.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, AsyncIterable
from pathlib import Path
from typing import Any

from strands.models import Model
from strands.models.litellm import LiteLLMModel
from strands.types.content import Messages
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "model_id": "openai/gpt-4o-mini",
        "key_env": "OPENAI_API_KEY",
        "key_hint": (
            "Get one at https://platform.openai.com/api-keys — the API is "
            "billed separately from a ChatGPT subscription."
        ),
    },
    "gemini": {
        "model_id": "gemini/gemini-3.5-flash-lite",
        "key_env": "GEMINI_API_KEY",
        "key_hint": (
            "Get a free key at Google AI Studio "
            "(https://aistudio.google.com) — no credit card needed."
        ),
    },
}

# Approximate input prices per 1M tokens, for cost estimates in traces.
# These move — check the provider pricing page before budgeting on them.
INPUT_PRICE_PER_MILLION: dict[str, float] = {
    "openai/gpt-4o-mini": 0.15,
    "gemini/gemini-3.5-flash-lite": 0.0,  # free tier: $0
}


def load_dotenv(env_path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file into the environment.

    Only sets keys that are not already set — the real environment always
    wins. Looks at ``.env`` in the working directory unless a path is
    given. Lines starting with ``#`` and blank lines are skipped, and
    surrounding quotes are stripped.
    """
    path = env_path or Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def require_provider_key(provider: str = "openai") -> str:
    """Return the API key for a provider, or fail loudly if missing."""
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose one of: "
            + ", ".join(sorted(PROVIDERS))
        )
    load_dotenv()
    key_env = PROVIDERS[provider]["key_env"]
    key = os.environ.get(key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"{key_env} is not set. The agent runs against real "
            f"{provider} by default — set the key in your environment, or "
            f"copy .env.example to .env and put the key there (never commit "
            f".env). {PROVIDERS[provider]['key_hint']} For the offline demo "
            f"only, run with --mock."
        )
    return key


def build_model(provider: str) -> LiteLLMModel:
    """Build the real LiteLLM model for a provider (key required)."""
    require_provider_key(provider)
    return LiteLLMModel(model_id=PROVIDERS[provider]["model_id"])


def fallback_provider_for(provider: str) -> str | None:
    """The other provider, for model fallback. None when no key exists."""
    other = "gemini" if provider == "openai" else "openai"
    load_dotenv()
    if os.environ.get(PROVIDERS[other]["key_env"], "").strip():
        return other
    return None


class ModelWithFallback(Model):
    """A Strands model that falls back to a second provider on failure.

    Primary first; on any error before the first content is produced,
    retry the same turn on the fallback model. The fallback is used at
    most once per turn. After content has started streaming, the turn
    cannot be restarted, so the error propagates instead.
    """

    def __init__(self, primary: Model, fallback: Model, primary_id: str = "", fallback_id: str = "") -> None:
        self._primary = primary
        self._fallback = fallback
        self._primary_id = primary_id
        self._fallback_id = fallback_id
        self._config: dict[str, Any] = {
            "model_id": f"{primary_id}+fallback:{fallback_id}"
        }
        self.fell_back = False

    def get_config(self) -> Any:
        return dict(self._config)

    def update_config(self, **model_config: Any) -> None:
        self._config.update(model_config)

    def structured_output(self, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        raise NotImplementedError("ModelWithFallback does not support structured output.")

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        try:
            produced = False
            async for event in self._primary.stream(
                messages, tool_specs, system_prompt, tool_choice=tool_choice, **kwargs
            ):
                produced = True
                yield event
            return
        except Exception:
            if produced:
                raise
        self.fell_back = True
        async for event in self._fallback.stream(
            messages, tool_specs, system_prompt, tool_choice=tool_choice, **kwargs
        ):
            yield event


def build_model_with_fallback(provider: str) -> Model:
    """Build the model for a provider, with fallback to the other provider
    when its key is available. No fallback when the other key is missing."""
    primary = build_model(provider)
    other = fallback_provider_for(provider)
    if other is None:
        return primary
    return ModelWithFallback(
        primary,
        build_model(other),
        primary_id=PROVIDERS[provider]["model_id"],
        fallback_id=PROVIDERS[other]["model_id"],
    )
