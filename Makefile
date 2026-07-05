PYTHON ?= python3
UV ?= uv
VENV_PYTHON ?= .venv/bin/python

.PHONY: sync compile pytest

sync:
	$(UV) sync --group dev

compile:
	$(PYTHON) -m compileall apps packages mcp_servers tests

pytest:
	$(VENV_PYTHON) -m pytest
