.PHONY: setup install format lint typecheck test run-brain run-satellite check

setup: install
	pre-commit install

install:
	pip install -e ".[dev]"

format:
	ruff format .
	ruff check --fix .

lint:
	ruff check .

typecheck:
	mypy --strict src

test:
	pytest

run-brain:
	./.venv/bin/uvicorn api:app --reload --host 0.0.0.0 --port 8000 --log-level info

run-satellite:
	python satelite.py

check: lint typecheck test
