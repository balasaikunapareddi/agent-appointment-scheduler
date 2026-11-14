"""Live tests — skipped without keys. Run manually once keys exist.

These are the only tests that touch a real model. They are skipped in
CI and in the offline suite; run them with OPENAI_API_KEY or
GEMINI_API_KEY set.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"),
    reason="no live model key set",
)


def test_live_demo_smoke(db_path):
    from agent_appointment_scheduler.chat import ChatService

    provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "openai"
    service = ChatService(db_path=db_path, provider=provider, mock=False)
    reply = service.chat("SES-LIVE1", "Find me a cardiologist near Minneapolis")
    assert reply.reply, "live model returned an empty reply"
    assert "search_providers" in reply.tools_used
    service.close()
