"""Multi-turn demo: one session, find -> book -> reschedule -> resume.

Starts the FastAPI service (and the chat UI) on localhost, then runs a
realistic conversation through the agent:

1. "I need a cardiologist near Minneapolis next week, I prefer mornings"
2. "the second one"
3. "Wednesday morning works"
4. "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147"
5. "Actually my phone is 612-555-0199"          (correction mid-flow)
6. "It's a heart checkup follow-up, text me"
7. "Yes, book it"                                (masked summary, then confirm)
8. "Actually, can we move it to Thursday morning instead?"
9. A FRESH agent instance resumes the same session:
   "What's my appointment?"

``make demo`` runs the real OpenAI model (needs OPENAI_API_KEY),
``make demo-gemini`` the Gemini free tier (needs GEMINI_API_KEY),
``make demo-mock`` the scripted model (no key, offline).

By default the UI stays up after the demo (``--serve``); pass
``--no-serve`` to stop the server and exit, for automation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import httpx

from .chat import ChatService
from .models import PROVIDERS


def _wait_for_health(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    # trust_env=False: localhost needs no proxy, and it avoids httpx
    # failing to parse a malformed no_proxy list in some sandboxes.
    with httpx.Client(trust_env=False, timeout=2.0) as client:
        while time.time() < deadline:
            try:
                response = client.get(url)
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
    return False


TURNS = [
    "I need a cardiologist near Minneapolis next week, I prefer mornings",
    "the second one",
    "Wednesday morning works",
    "My name is Jordan Lee, born May 14, 1990, phone 612-555-0147",
    "Actually my phone is 612-555-0199",
    "It's a heart checkup follow-up, text me",
    "Yes, book it",
    "Actually, can we move it to Thursday morning instead?",
]


def run_demo(provider: str = "openai", mock: bool = False, serve: bool = True, port: int = 8000) -> int:
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_file.close()
    db_path = db_file.name
    env = {
        **os.environ,
        "SCHEDULER_DB_PATH": db_path,
        "SCHEDULER_API_PORT": str(port),
        "SCHEDULER_API_URL": f"http://127.0.0.1:{port}",
        "SCHEDULER_PROVIDER": provider,
    }
    if mock:
        env["SCHEDULER_MOCK"] = "1"
    # The MCP server subprocess inherits this process's environment, so
    # the database path and API URL must be set here too, not just for
    # the uvicorn subprocess below.
    os.environ["SCHEDULER_DB_PATH"] = db_path
    os.environ["SCHEDULER_API_URL"] = f"http://127.0.0.1:{port}"
    os.environ["SCHEDULER_PROVIDER"] = provider
    if mock:
        os.environ["SCHEDULER_MOCK"] = "1"
    print(f"Starting the scheduling service on http://127.0.0.1:{port} ...")
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "agent_appointment_scheduler.api:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env,
    )
    try:
        if not _wait_for_health(f"http://127.0.0.1:{port}/health"):
            print("ERROR: the service did not answer /health in time.")
            return 1
        print(f"Service is up. Chat UI: http://127.0.0.1:{port}/")
        session_id = "SES-DEMO1"
        print(f"\n--- Session {session_id} ({'scripted mock' if mock else PROVIDERS[provider]['model_id']}) ---")
        service = ChatService(db_path=db_path, provider=provider, mock=mock)
        for turn in TURNS:
            print(f"\nYou: {turn}")
            reply = service.chat(session_id, turn)
            print(f"Agent: {reply.reply}")
            if reply.tools_used:
                print(f"  [tools: {', '.join(reply.tools_used)} | stage: {reply.booking_stage}]")
            if reply.cache:
                cache = reply.cache
                print(
                    f"  [tokens: prompt={cache.get('prompt_tokens', 0)} "
                    f"cached={cache.get('cached_tokens', 0)} "
                    f"hit_rate={cache.get('cache_hit_rate', 0)} "
                    f"est_saved=${cache.get('est_cost_saved_usd', 0)}]"
                )
        service.close()
        # Resume: a fresh agent instance, same session id.
        print("\n--- Resuming the same session in a fresh agent instance ---")
        resumed = ChatService(db_path=db_path, provider=provider, mock=mock)
        reply = resumed.chat(session_id, "What's my appointment?")
        print(f"\nYou: What's my appointment?")
        print(f"Agent: {reply.reply}")
        print(f"  [tools: {', '.join(reply.tools_used)} | stage: {reply.booking_stage}]")
        resumed.close()
        print("\nDemo complete.")
        if serve:
            print(f"The UI is still running at http://127.0.0.1:{port}/ — press Ctrl-C to stop.")
            try:
                server.wait()
            except KeyboardInterrupt:
                print("\nStopping the service.")
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multi-turn scheduling demo.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mock", action="store_true", help="offline scripted model (no key)")
    group.add_argument("--gemini", action="store_true", help="Gemini free tier")
    group.add_argument("--live", action="store_true", help="real OpenAI model (default)")
    parser.add_argument("--provider", default="openai", choices=list(PROVIDERS), help="model provider")
    parser.add_argument("--no-serve", action="store_true", help="stop the server after the demo")
    parser.add_argument("--port", type=int, default=8000, help="API/UI port")
    args = parser.parse_args()
    provider = "gemini" if args.gemini else args.provider
    return run_demo(provider=provider, mock=args.mock, serve=not args.no_serve, port=args.port)


if __name__ == "__main__":
    sys.exit(main())
