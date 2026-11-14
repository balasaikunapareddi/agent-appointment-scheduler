.PHONY: install test demo demo-gemini demo-mock demo-live server eval

VENV ?= .venv
PY := $(VENV)/bin/python

install:
	python3 -m venv $(VENV) || true
	$(VENV)/bin/pip install -q -U pip
	$(VENV)/bin/pip install -q -e ".[dev]"

test:
	$(PY) -m pytest -q

# Real OpenAI model (needs OPENAI_API_KEY). Serves the UI until Ctrl-C.
demo:
	$(PY) -m agent_appointment_scheduler.demo --live

demo-gemini:
	$(PY) -m agent_appointment_scheduler.demo --gemini

demo-mock:
	$(PY) -m agent_appointment_scheduler.demo --mock --no-serve

demo-live:
	$(PY) -m agent_appointment_scheduler.demo --live --no-serve

# Just the service + UI (mock model for /chat unless SCHEDULER_MOCK is unset).
server:
	$(PY) -m agent_appointment_scheduler.api

eval:
	$(PY) -m agent_appointment_scheduler.eval
