"""Turn-level rate-limit retries: a 429 must not kill a turn.

The live Gemini demo proved this: the free tier allows 15 requests/min,
and a 429 landing mid-stream surfaced as a MidStreamFallbackError that
crashed the whole demo run. ChatService.chat now retries the turn after
a backoff instead of raising.
"""

from types import SimpleNamespace

import pytest
from litellm.exceptions import RateLimitError

from agent_appointment_scheduler import chat as chat_module
from agent_appointment_scheduler.chat import RATE_LIMIT_RETRY_DELAYS, ChatService


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(
        "429 quota exceeded for gemini-3.5-flash-lite",
        llm_provider="gemini",
        model="gemini-3.5-flash-lite",
    )


class _FakeHooks:
    last_metrics = None

    def begin_turn(self):
        pass

    def end_turn(self, message, reply, trace_id=None):
        return {"tools": []}


class _FakeSkills:
    def get_activated_skills(self, agent):
        return []


class _FakeAgent:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def __call__(self, message):
        self.calls += 1
        if self.calls <= self.failures:
            raise _rate_limit_error()
        return SimpleNamespace(message={"content": [{"text": "booked"}]})


@pytest.fixture()
def _service(db_path, monkeypatch):
    service = ChatService(db_path=db_path, provider="gemini", mock=True)
    sleeps = []
    monkeypatch.setattr(chat_module.time, "sleep", lambda s: sleeps.append(s))
    return service, sleeps


def _patch_agent(service, monkeypatch, agent):
    monkeypatch.setattr(
        ChatService, "_get_agent", lambda self, sid: (agent, (None, _FakeHooks(), _FakeSkills()))
    )


def test_chat_retries_once_on_rate_limit(_service, monkeypatch):
    service, sleeps = _service
    agent = _FakeAgent(failures=1)
    _patch_agent(service, monkeypatch, agent)
    reply = service.chat("SES-TEST", "book it")
    assert reply.reply == "booked"
    assert agent.calls == 2
    assert sleeps == [RATE_LIMIT_RETRY_DELAYS[0]]


def test_chat_gives_up_after_max_retries(_service, monkeypatch):
    service, sleeps = _service
    agent = _FakeAgent(failures=99)
    _patch_agent(service, monkeypatch, agent)
    with pytest.raises(RateLimitError):
        service.chat("SES-TEST", "book it")
    assert agent.calls == len(RATE_LIMIT_RETRY_DELAYS) + 1
    assert sleeps == list(RATE_LIMIT_RETRY_DELAYS)


def test_chat_does_not_retry_other_errors(_service, monkeypatch):
    service, sleeps = _service

    class _BoomAgent:
        calls = 0

        def __call__(self, message):
            type(self).calls += 1
            raise RuntimeError("primary provider is down")

    _patch_agent(service, monkeypatch, _BoomAgent())
    with pytest.raises(RuntimeError, match="primary provider is down"):
        service.chat("SES-TEST", "book it")
    assert _BoomAgent.calls == 1
    assert sleeps == []
