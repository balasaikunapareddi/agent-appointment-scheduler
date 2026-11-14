"""Demo CLI test: the mock demo runs end to end and exits.

This starts the real FastAPI service on a free port, runs the full
multi-turn demo through the scripted model, and asserts it exits 0
with the booking and reschedule turns in the output.
"""

import socket
import subprocess
import sys


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_demo_mock_completes():
    port = _free_port()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_appointment_scheduler.demo",
            "--mock",
            "--no-serve",
            "--port",
            str(port),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Booked!" in result.stdout
    assert "Moved it" in result.stdout
    assert "What's my appointment?" in result.stdout
    assert f"http://127.0.0.1:{port}/" in result.stdout
