"""Session + memory tests: resume, summarization, preferences, token budget."""

from agent_appointment_scheduler.sessions import (
    SessionStore,
    SQLiteSessionRepository,
    estimate_tokens,
)
from agent_appointment_scheduler.slots import BookingForm


def test_create_and_list_sessions(db_path):
    store = SessionStore(db_path)
    sid = store.create_session(label="demo")
    assert store.get_session(sid)["label"] == "demo"
    assert len(store.list_sessions()) == 1


def test_history_is_redacted(db_path):
    store = SessionStore(db_path)
    store.register_sensitive_values("Jordan Lee", "612-555-0147")
    store.append_message("SES-1", "user", "My name is Jordan Lee, phone 612-555-0147")
    history = store.history("SES-1")
    assert len(history) == 1
    assert "Jordan Lee" not in history[0]["content"]
    assert "612-555-0147" not in history[0]["content"]


def test_preferences_recorded_and_recalled(db_path):
    store = SessionStore(db_path)
    found = store.record_preferences_from_text("SES-1", "I prefer mornings near Minneapolis, text me")
    assert found["time_of_day"] == "morning"
    assert found["city"] == "Minneapolis"
    assert found["contact_method"] == "text"
    prefs = store.get_preferences("SES-1")
    assert prefs["time_of_day"] == "morning"


def test_summarization_trigger(db_path):
    store = SessionStore(db_path)
    for i in range(14):
        store.append_message("SES-1", "user" if i % 2 == 0 else "assistant", f"turn {i} about cardiology")
    assert store.maybe_summarize("SES-1") is True
    summary = store.get_summary("SES-1")
    assert summary, "summary should be stored after the threshold"
    assert "turn 0" in summary or "cardiology" in summary


def test_no_summarization_below_threshold(db_path):
    store = SessionStore(db_path)
    store.append_message("SES-1", "user", "hi")
    assert store.maybe_summarize("SES-1") is False


def test_build_context_respects_budget(db_path):
    store = SessionStore(db_path)
    for i in range(20):
        store.append_message("SES-1", "user", "x" * 500 + f" turn {i}")
    context = store.build_context("SES-1", budget_tokens=1000)
    assert context["estimated_tokens"] <= 1000 + 500  # last message may push slightly over
    assert context["total_messages"] == 20
    assert len(context["messages"]) < 20


def test_booking_form_persists(db_path):
    store = SessionStore(db_path)
    form = BookingForm()
    form.provider_npi = "1215596358"
    form.provider_name = "Muhannad Abbasi"
    form.stage = "providers_shown"
    store.save_booking_form("SES-1", form)
    loaded = store.get_booking_form("SES-1")
    assert loaded.provider_npi == "1215596358"
    assert loaded.stage == "providers_shown"


def test_estimate_tokens():
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 400) == 100


def test_repository_round_trip(db_path):
    from strands.types.session import Session, SessionAgent, SessionMessage

    repo = SQLiteSessionRepository(db_path)
    session = repo.create_session(Session(session_id="S1", session_type="test", created_at="t", updated_at="t"))
    assert session.session_id == "S1"
    assert repo.read_session("S1").session_id == "S1"
    assert repo.read_session("missing") is None
    message = SessionMessage(
        message={"role": "user", "content": [{"text": "hello"}]},
        message_id=1,
        redact_message=None,
        created_at="t",
        updated_at="t",
    )
    repo.create_message("S1", "A1", message)
    listed = repo.list_messages("S1", "A1")
    assert len(listed) == 1
    assert listed[0].message["content"][0]["text"] == "hello"
    agent = SessionAgent(
        agent_id="A1", state={}, conversation_manager_state={}, _internal_state={}, created_at="t", updated_at="t"
    )
    repo.create_agent("S1", agent)
    assert repo.read_agent("S1", "A1").agent_id == "A1"
