"""MCP tool tests: call the server tools directly over ASGI transport."""

from agent_appointment_scheduler import mcp_server


def test_search_providers_tool(db_path):
    result = mcp_server.search_providers("cardiology", city="Minneapolis", state="MN", session_id="SES-M1")
    assert result["ok"] is True
    assert result["providers"]


def test_get_availability_tool(db_path):
    providers = mcp_server.search_providers("cardiology", city="Minneapolis", state="MN", session_id="SES-M1")
    npi = providers["providers"][0]["npi"]
    result = mcp_server.get_availability(npi, date_from="2026-11-16", date_to="2026-11-20", time_of_day="morning")
    assert result["ok"] is True
    assert result["slots"]
    assert all(s["slot"][11:13] < "12" for s in result["slots"] if s["available"])


def test_get_plan_tool(db_path):
    plan = mcp_server.get_plan("SES-M1")
    assert plan["ok"] is True
    assert plan["plan"]["steps"][0]["tool"] == "search_providers"


def test_booking_status_tool_empty(db_path):
    status = mcp_server.get_booking_status("SES-M1")
    assert status["ok"] is True
    assert status["missing"], "a fresh session must have missing slots"
    assert status["next_question"]


def test_create_patient_tool_masks(db_path):
    result = mcp_server.create_patient_record(
        "Jordan Lee", "1990-05-14", "612-555-0147", session_id="SES-M1"
    )
    assert result["ok"] is True
    assert result["patient_ref"].startswith("PAT-")
    assert result["phone_masked"] == "***-***-0147"
    assert "612-555-0147" not in str(result)
