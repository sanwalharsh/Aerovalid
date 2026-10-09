.PHONY: setup lint test test-fast demo dashboard config-ref

VENV := .venv
PY := $(VENV)/bin/python
BIN := $(VENV)/bin

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[all]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy src

test:
	$(BIN)/pytest -q --cov=aerovalid --cov-fail-under=85

test-fast:
	$(BIN)/pytest -q -m "not slow"

demo:
	$(BIN)/fsa run-all --config configs/default.yaml --profile demo

dashboard:
	$(BIN)/streamlit run src/aerovalid/app/main.py

config-ref:
	$(PY) scripts/gen_config_reference.py
