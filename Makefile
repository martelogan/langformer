SHELL := /bin/bash

.PHONY: setup lint test typecheck pylint

setup:
	uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

lint:
	uv run ruff check --select I --fix
	uv run ruff check .

test:
	pytest -q

typecheck:
	uv run pyright

pylint:
	uv run pylint --errors-only langformer
